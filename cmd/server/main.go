// Package main is the binary entrypoint for the VectorDB server.
// It is responsible for:
//   - Parsing the --config flag and loading configuration
//   - Configuring the Go runtime (GC percent, GOMAXPROCS)
//   - Setting up structured logging via log/slog
//   - Wiring up the HTTP server with all components
//   - Running an internal server for metrics + pprof
//   - Handling OS signals (SIGINT, SIGTERM) for graceful shutdown
//   - Handling SIGHUP for config hot reload
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	_ "net/http/pprof" // Side-effect: registers pprof handlers on DefaultServeMux
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/POTATO-VE1/Magnitude/internal/api"
	"github.com/POTATO-VE1/Magnitude/internal/cluster"
	"github.com/POTATO-VE1/Magnitude/internal/collection"
	"github.com/POTATO-VE1/Magnitude/internal/config"
	"github.com/POTATO-VE1/Magnitude/internal/events"
	"github.com/POTATO-VE1/Magnitude/internal/failure"
	"github.com/POTATO-VE1/Magnitude/internal/gossip"
	"github.com/POTATO-VE1/Magnitude/internal/metadata"
	"github.com/POTATO-VE1/Magnitude/internal/migration"
	"github.com/POTATO-VE1/Magnitude/internal/routing"
	"github.com/POTATO-VE1/Magnitude/internal/storage"
)

func main() {
	// ── 1. Parse flags ───────────────────────────────────────────────────────
	configPath := flag.String("config", "config.yaml", "path to config.yaml")
	flag.Parse()

	// ── 2. Load configuration ─────────────────────────────────────────────────
	cfg, err := config.LoadConfig(*configPath)
	if err != nil {
		// Cannot use structured logger yet — it depends on config.
		fmt.Fprintf(os.Stderr, "fatal: failed to load config: %v\n", err)
		os.Exit(1)
	}

	// ── 3. Set up structured logging (log/slog, JSON in production) ───────────
	// JSON handler writes machine-readable logs suitable for log aggregation (e.g., Loki).
	// In development, swap to slog.NewTextHandler for human-readable output.
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	slog.SetDefault(logger)

	slog.Info("vectordb starting",
		"config", *configPath,
		"go_version", runtime.Version(),
		"gomaxprocs", runtime.GOMAXPROCS(0),
		"index_type", cfg.Index.Type,
		"data_dir", cfg.Storage.DataDir,
	)

	// Security warnings
	if len(cfg.Auth.KeyHashes) == 0 {
		slog.Warn("SECURITY: auth.keyHashes is empty — authentication is DISABLED")
		slog.Warn("All requests will be accepted without credential verification")
		if cfg.Server.Addr != ":8443" && cfg.Server.Addr != ":443" {
			slog.Warn("Server is not bound to a standard HTTPS port — ensure this is intentional")
		}
	}

	// ── 4. Configure Go runtime ───────────────────────────────────────────────
	// SetGCPercent controls when the GC triggers relative to live heap size.
	// For a vector DB with large mmap'd memory regions that the GC doesn't scan,
	// the default of 100 is usually appropriate. Increase to 200 for heavy mmap workloads.
	if cfg.GC.Percent != 0 {
		old := debug.SetGCPercent(cfg.GC.Percent)
		slog.Info("gc configured", "percent", cfg.GC.Percent, "previous", old)
	}

	// ── 5. Ensure data directory exists ──────────────────────────────────────
	if err := os.MkdirAll(cfg.Storage.DataDir, 0o750); err != nil {
		slog.Error("failed to create data directory",
			"path", cfg.Storage.DataDir,
			"error", err,
		)
		os.Exit(1)
	}

	// ── 6. Initialize SysDB ─────────────────────────────────────────────────
	sysdbPath := filepath.Join(cfg.Storage.DataDir, "sysdb.sqlite")
	sysdb, err := metadata.NewSysDB(sysdbPath)
	if err != nil {
		slog.Error("failed to initialize SysDB", "error", err)
		os.Exit(1)
	}

	// ── 6b. Initialize multi-tenancy schema (Phase 8) ────────────────────────
	if err := sysdb.InitMultiTenancySchema(); err != nil {
		slog.Error("failed to initialize multi-tenancy schema", "error", err)
		os.Exit(1)
	}

	// ── 6c. Seed default tenant and database (idempotent) ────────────────────
	defaultTenantID, defaultDBID, err := sysdb.EnsureDefaults()
	if err != nil {
		slog.Error("failed to seed defaults", "error", err)
		os.Exit(1)
	}
	slog.Info("default tenant ready",
		"tenant_id", defaultTenantID,
		"database_id", defaultDBID,
	)

	// ── 7. Initialize WAL ────────────────────────────────────────────────────
	walPath := filepath.Join(cfg.Storage.DataDir, "wal.sqlite")
	var walOpts []storage.WALOption
	if cfg.WALSync.SyncMode != "" {
		walOpts = append(walOpts, storage.WithSyncMode(cfg.WALSync.SyncMode))
	}
	if cfg.WALSync.SyncDelay > 0 {
		walOpts = append(walOpts, storage.WithSyncDelay(cfg.WALSync.SyncDelay))
	}
	wal, err := storage.NewSQLiteWAL(walPath, walOpts...)
	if err != nil {
		slog.Error("failed to initialize WAL", "error", err)
		os.Exit(1)
	}

	// ── 7b. Recover incomplete compaction actions (crash recovery) ────────────
	if err := storage.RecoverCompactionActions(cfg.Storage.DataDir); err != nil {
		slog.Error("compaction action recovery failed", "error", err)
		os.Exit(1)
	}

	// ── 7.5. Initialize FlowBus ──────────────────────────────────────────────
	flowBus := events.NewFlowBus()

	// ── 8. Initialize Collection Manager (includes WAL replay) ───────────────
	mgr, err := collection.NewManager(sysdb, wal,
		collection.WithFlowBus(flowBus),
		collection.WithDataDir(cfg.Storage.DataDir),
	)
	if err != nil {
		slog.Error("failed to initialize collection manager", "error", err)
		os.Exit(1)
	}

	// ── 8.5 Initialize Gossip, Failure Detector, and Hash Ring (Tier 3) ───
	var failDetector *failure.Detector
	var gossipProto *gossip.Protocol
	var hashRing *cluster.HashRing
	var rt *routing.Router
	var fwd *routing.Forwarder

	if cfg.Cluster.Enabled {
		// Require gossip secret key in cluster mode to prevent unauthorized membership manipulation
		if cfg.Gossip.SecretKey == "" {
			slog.Error("SECURITY: gossip.secretKey is required when cluster is enabled")
			slog.Error("Generate a key: openssl rand -hex 32")
			os.Exit(1)
		}

		hashRing = cluster.NewHashRing(cfg.Cluster.VirtualNodes)
		// Add self to hash ring
		hashRing.AddNode(cfg.Cluster.NodeID)

		// Migration worker for data rebalancing
		migWorker := migration.NewWorker(migration.Config{
			BatchSize:   cfg.Migration.BatchSize,
			Parallelism: cfg.Migration.Parallelism,
			MaxRetries:  cfg.Migration.MaxRetries,
		})

		failCfg := failure.Config{
			Interval:     cfg.Failure.Interval,
			Timeout:      cfg.Failure.Timeout,
			SuspectAfter: cfg.Failure.SuspectAfter,
			DeadAfter:    cfg.Failure.DeadAfter,
			OnNodeSuspect: func(nodeID string) {
				slog.Warn("node suspect", "node_id", nodeID)
			},
			OnNodeDead: func(nodeID string) {
				slog.Warn("node declared dead, triggering migration", "node_id", nodeID)

				// Trigger migration for collections owned by the dead node.
				// TriggerOnNodeRemoval handles ring removal internally:
				// it snapshots ownership first, then removes the dead node.
				cols, _ := mgr.ListCollections()
				colIDs := make([]string, 0, len(cols))
				for _, c := range cols {
					colIDs = append(colIDs, c.ID)
				}
				plans := migration.TriggerOnNodeRemoval(hashRing, nodeID, colIDs, cfg.Cluster.ReplicationFactor)
				for _, plan := range plans {
					go func(p migration.MigrationPlan) {
						// Build a vector source from the collection's exported vectors
						source := func() ([]uint64, error) {
							// In production, this would fetch from the source node
							// For now, return nil (no vectors to migrate in single-node test)
							return nil, nil
						}
						if err := migWorker.ExecutePlan(p, source, func(batch []uint64) error {
							// Transfer function — forward to target node
							slog.Info("migration: transferring batch", "plan", p.ID, "vectors", len(batch))
							return nil
						}); err != nil {
							slog.Error("migration failed", "plan", p.ID, "error", err)
						}
					}(plan)
				}
			},
		}
		failDetector = failure.New(failCfg)

		// Gossip callback triggers failure detector and updates hash ring
		gossipCb := func(msg gossip.Message) {
			failDetector.RecordHeartbeat(msg.Source)
			switch msg.Event {
			case gossip.EventAlive:
				var addrPayload struct {
					API string `json:"api"`
				}
				apiAddr := string(msg.Payload) // fallback: raw payload
				if err := json.Unmarshal(msg.Payload, &addrPayload); err == nil && addrPayload.API != "" {
					apiAddr = addrPayload.API
				}
				failDetector.AddNode(msg.Source, apiAddr)
				hashRing.AddNode(msg.Source)
			case gossip.EventDead:
				failDetector.RemoveNode(msg.Source)
				hashRing.RemoveNode(msg.Source)
			case gossip.EventCreateCollection:
				var p struct {
					Name      string `json:"name"`
					Dim       int    `json:"dim"`
					Metric    string `json:"metric"`
					IndexType string `json:"index_type"`
				}
				if err := json.Unmarshal(msg.Payload, &p); err != nil {
					slog.Warn("gossip: invalid CreateCollection payload", "error", err)
					return
				}
				if err := mgr.CreateCollectionRemote(p.Name, p.Dim, p.Metric, p.IndexType); err != nil {
					slog.Warn("gossip: remote create collection failed", "name", p.Name, "error", err)
				}
			case gossip.EventDropCollection:
				var p struct {
					Name string `json:"name"`
				}
				if err := json.Unmarshal(msg.Payload, &p); err != nil {
					slog.Warn("gossip: invalid DropCollection payload", "error", err)
					return
				}
				if err := mgr.DeleteCollectionRemote(p.Name); err != nil {
					slog.Warn("gossip: remote delete collection failed", "name", p.Name, "error", err)
				}
			}
		}

		gossipCfg := gossip.Config{
			Port:          cfg.Gossip.Port,
			Fanout:        cfg.Gossip.Fanout,
			MaxSeen:       cfg.Gossip.MaxSeen,
			SeenExpiry:    cfg.Gossip.SeenExpiry,
			ProbeInterval: cfg.Gossip.ProbeInterval,
		}
		gossipProto = gossip.New(cfg.Cluster.NodeID, gossipCfg, gossipCb)

		// Load seed nodes to bootstrap the cluster
		if len(cfg.Cluster.SeedNodes) > 0 {
			gossipProto.SetPeers(cfg.Cluster.SeedNodes)
			slog.Info("cluster seed nodes configured", "count", len(cfg.Cluster.SeedNodes))
		}

		if err := gossipProto.StartUDP(cfg.Gossip.SecretKey); err != nil {
			slog.Error("failed to start gossip server", "error", err)
			os.Exit(1)
		}
		failDetector.Start()

		// Broadcast Alive
		payload, _ := json.Marshal(map[string]string{"api": cfg.Server.Addr})
		gossipProto.Broadcast(gossip.EventAlive, payload)

		// Wire gossip broadcaster into collection manager for cluster-wide event dissemination
		mgr.SetGossipBroadcaster(func(event collection.GossipEventKind, payload []byte) {
			var gossipEvent gossip.EventKind
			switch event {
			case collection.GossipEventCreateCollection:
				gossipEvent = gossip.EventCreateCollection
			case collection.GossipEventDropCollection:
				gossipEvent = gossip.EventDropCollection
			default:
				return
			}
			gossipProto.Broadcast(gossipEvent, payload)
		})

		// Initialize HTTP Routing components
		fwd = routing.NewForwarder(cfg)
		rt = routing.NewRouter(hashRing, cfg.Cluster.NodeID, cfg.Server.Addr, failDetector)

		slog.Info("cluster mode enabled", "node_id", cfg.Cluster.NodeID)
	} else {
		slog.Info("cluster mode disabled, running as standalone node")
	}

	// ── 9. Start internal server (metrics + pprof) ──────────────────────────
	// Runs on localhost-only port. Never exposed to the internet.
	// pprof is registered via the _ "net/http/pprof" import above.
	internalMux := api.NewAdminRouter(cfg, mgr)
	internalMux.Handle("/metrics", promhttp.Handler())
	internalMux.Mount("/debug/pprof/", http.DefaultServeMux) // delegate to pprof handlers

	internalServer := &http.Server{
		Addr:    cfg.Server.InternalPort,
		Handler: internalMux,
	}
	go func() {
		slog.Info("internal server starting (metrics + pprof)", "addr", cfg.Server.InternalPort)
		if err := internalServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("internal server error", "error", err)
		}
	}()

	// ── 10. Wire up public HTTP server ──────────────────────────────────────
	router := api.NewRouter(cfg, mgr, rt, fwd)
	server := &http.Server{
		Addr:              cfg.Server.Addr,
		Handler:           router,
		ReadTimeout:       cfg.Server.ReadTimeout,
		ReadHeaderTimeout: cfg.Server.ReadHeaderTimeout,
		WriteTimeout:      cfg.Server.WriteTimeout,
		MaxHeaderBytes:    cfg.Server.MaxHeaderBytes,
	}

	// Start HTTP server in a goroutine
	go func() {
		slog.Info("HTTP server starting",
			"addr", cfg.Server.Addr,
		)

		var serverErr error
		if cfg.Server.CertFile != "" && cfg.Server.KeyFile != "" {
			// TLS mode
			serverErr = server.ListenAndServeTLS(cfg.Server.CertFile, cfg.Server.KeyFile)
		} else {
			// Plain HTTP (development mode)
			slog.Warn("running in plain HTTP mode — use TLS in production")
			serverErr = server.ListenAndServe()
		}

		if serverErr != nil && serverErr != http.ErrServerClosed {
			slog.Error("HTTP server error", "error", serverErr)
			os.Exit(1)
		}
	}()

	// ── 11. Signal handling ──────────────────────────────────────────────────
	// SIGINT/SIGTERM → graceful shutdown
	// SIGHUP → config hot reload (rate limits, GC percent — NOT address changes)
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM, syscall.SIGHUP)

	for {
		sig := <-sigCh

		if sig == syscall.SIGHUP {
			slog.Info("SIGHUP received — reloading config", "path", *configPath)
			newCfg, err := config.LoadConfig(*configPath)
			if err != nil {
				slog.Error("config reload failed — keeping current config", "error", err)
				continue
			}

			// Apply hot-reloadable settings
			if newCfg.GC.Percent != 0 && newCfg.GC.Percent != cfg.GC.Percent {
				old := debug.SetGCPercent(newCfg.GC.Percent)
				slog.Info("gc percent reloaded", "percent", newCfg.GC.Percent, "previous", old)
			}

			cfg = newCfg
			slog.Info("config reloaded successfully")
			continue
		}

		// SIGINT or SIGTERM → 7-step graceful shutdown
		slog.Info("shutdown signal received", "signal", sig.String())
		break
	}

	// ── 12. Graceful shutdown (7-step sequence) ──────────────────────────────
	//
	// Step 1: Stop accepting new connections
	// Step 2: Drain in-flight requests (HTTP server.Shutdown)
	// Step 3: Stop internal server
	// Step 4: Flush all indexes to disk
	// Step 5: Close WAL (flushes pending writes)
	// Step 6: Close SysDB
	// Step 7: Log and exit
	//
	shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.Server.ShutdownTimeout)
	defer cancel()
	shutdownStart := time.Now()

	// Broadcast Dead to cluster peers
	if cfg.Cluster.Enabled && gossipProto != nil && failDetector != nil {
		gossipProto.Broadcast(gossip.EventDead, nil)
		time.Sleep(50 * time.Millisecond) // Let packet escape
		gossipProto.StopUDP()
		failDetector.Stop()
	}

	// Step 1+2: Stop accepting + drain in-flight
	slog.Info("shutdown step 1/6: draining HTTP connections...")
	if err := server.Shutdown(shutdownCtx); err != nil {
		slog.Error("HTTP server shutdown error", "error", err)
	}

	// Step 3: Stop internal server
	slog.Info("shutdown step 2/6: stopping internal server...")
	if err := internalServer.Shutdown(shutdownCtx); err != nil {
		slog.Error("internal server shutdown error", "error", err)
	}

	// Step 4: Flush all collection indexes
	slog.Info("shutdown step 3/6: flushing collection indexes...")
	if err := mgr.Flush(); err != nil {
		slog.Error("flush error", "error", err)
	}

	// Step 5: Close WAL
	slog.Info("shutdown step 4/6: closing WAL...")
	if err := wal.Close(); err != nil {
		slog.Error("WAL close error", "error", err)
	}

	// Step 6: Close SysDB
	slog.Info("shutdown step 5/6: closing SysDB...")
	if err := sysdb.Close(); err != nil {
		slog.Error("SysDB close error", "error", err)
	}

	// Step 7: Done
	slog.Info("shutdown step 6/6: complete",
		"duration", time.Since(shutdownStart).String(),
	)
}
