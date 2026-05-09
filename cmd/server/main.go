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

	"github.com/veda/vectordb/internal/api"
	"github.com/veda/vectordb/internal/collection"
	"github.com/veda/vectordb/internal/config"
	"github.com/veda/vectordb/internal/events"
	"github.com/veda/vectordb/internal/metadata"
	"github.com/veda/vectordb/internal/storage"
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
	mgr, err := collection.NewManager(sysdb, wal, collection.WithFlowBus(flowBus))
	if err != nil {
		slog.Error("failed to initialize collection manager", "error", err)
		os.Exit(1)
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
	router := api.NewRouter(cfg, mgr)
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
