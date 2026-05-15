// Package config provides the Config struct and loading logic for the VectorDB server.
// All tunables are externalized here — no hardcoded values exist in production code.
// The config is loaded once at startup and passed through the system via dependency injection.
package config

import (
	"fmt"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

// Config is the root configuration struct for the VectorDB server.
// It mirrors config.yaml exactly. All nested structs use yaml tags for
// YAML parsing and json tags for future JSON config support.
type Config struct {
	Server      ServerConfig      `yaml:"server"`
	Auth        AuthConfig        `yaml:"auth"`
	RateLimit   RateLimitConfig   `yaml:"rateLimit"`
	Storage     StorageConfig     `yaml:"storage"`
	Index       IndexConfig       `yaml:"index"`
	GC          GCConfig          `yaml:"gc"`
	BloomFilter BloomFilterConfig `yaml:"bloomFilter"`
	WALSync     WALSyncConfig     `yaml:"walSync"`
	Cluster     ClusterConfig     `yaml:"cluster"`
	Gossip      GossipConfig      `yaml:"gossip"`
	Failure     FailureConfig     `yaml:"failureDetection"`
	Migration   MigrationConfig   `yaml:"migration"`
}

// ServerConfig holds all HTTP server tunables.
type ServerConfig struct {
	// Addr is the address the TLS server listens on. Default: :8443
	Addr string `yaml:"addr"`

	// InternalPort is the address for metrics and pprof endpoints (localhost only). Default: :9090
	InternalPort string `yaml:"internalPort"`

	// CertFile is the path to the TLS certificate file.
	CertFile string `yaml:"certFile"`

	// KeyFile is the path to the TLS private key file.
	KeyFile string `yaml:"keyFile"`

	// ReadTimeout is the maximum duration for reading the entire request. Default: 30s
	ReadTimeout time.Duration `yaml:"readTimeout"`

	// ReadHeaderTimeout is the maximum duration for reading request headers.
	// Set conservatively to protect against slow-loris attacks. Default: 5s
	ReadHeaderTimeout time.Duration `yaml:"readHeaderTimeout"`

	// WriteTimeout is the maximum duration before timing out writes of the response. Default: 30s
	WriteTimeout time.Duration `yaml:"writeTimeout"`

	// ShutdownTimeout is the maximum duration for graceful shutdown. Default: 30s
	ShutdownTimeout time.Duration `yaml:"shutdownTimeout"`

	// MaxHeaderBytes controls the maximum number of bytes read in request headers. Default: 16384 (16 KiB)
	MaxHeaderBytes int `yaml:"maxHeaderBytes"`
}

// AuthConfig holds authentication configuration.
// NEVER store raw API keys — only SHA-256 hashes (hex-encoded).
type AuthConfig struct {
	// KeyHashes is a list of SHA-256 hashes (hex-encoded) of valid API keys.
	// To add a key: echo -n "my-secret-key" | sha256sum
	KeyHashes []string `yaml:"keyHashes"`
}

// RateLimitConfig holds per-endpoint rate limiting configuration using token buckets.
type RateLimitConfig struct {
	// SearchRPS is the token bucket rate (requests/second) for search endpoints. Default: 100
	SearchRPS float64 `yaml:"searchRPS"`

	// SearchBurst is the maximum burst size for search requests. Default: 200
	SearchBurst int `yaml:"searchBurst"`

	// InsertRPS is the token bucket rate (requests/second) for insert endpoints. Default: 50
	InsertRPS float64 `yaml:"insertRPS"`

	// InsertBurst is the maximum burst size for insert requests. Default: 100
	InsertBurst int `yaml:"insertBurst"`
}

// StorageConfig holds persistence and memory-map configuration.
type StorageConfig struct {
	// DataDir is the root directory for all on-disk data (WAL, segments, SQLite). Default: ./data
	DataDir string `yaml:"dataDir"`

	// MmapCapacity is the maximum number of vectors to pre-allocate in the mmap file. Default: 1_000_000
	MmapCapacity int `yaml:"mmapCapacity"`
}

// IndexConfig holds algorithm-specific tuning parameters for all index types.
type IndexConfig struct {
	// Type selects the active index implementation: "flat", "ivf", or "hnsw".
	// Swapped in the collection constructor — the rest of the system is unaware.
	Type string `yaml:"type"`

	// NumClusters is the number of Voronoi cells for IVF indexing.
	// Rule of thumb: sqrt(N) where N is the expected vector count. Default: 256
	NumClusters int `yaml:"numClusters"`

	// DefaultNprobe is the default number of IVF clusters to search at query time.
	// Higher = better recall, slower search. Default: 5
	DefaultNprobe int `yaml:"defaultNprobe"`

	// M is the maximum number of bidirectional connections per node per HNSW layer.
	// Higher M = better recall + more memory. Typical range: 8–48. Default: 16
	M int `yaml:"m"`

	// EfConstruction is the HNSW beam width during index construction.
	// Higher = better graph quality + slower inserts. Must be > M. Default: 200
	EfConstruction int `yaml:"efConstruction"`

	// DefaultEf is the HNSW beam width during search (efSearch).
	// Higher = better recall + slower search. Default: 50
	DefaultEf int `yaml:"defaultEf"`

	// DirtyThreshold is the fraction of deleted vectors that triggers an automatic
	// Rebuild(). At 0.10, rebuilds when 10% of vectors have been soft-deleted.
	DirtyThreshold float64 `yaml:"dirtyThreshold"`

	// SnapshotInterval controls how often HNSW snapshots are taken.
	// 0 = only on shutdown. Default: 5 minutes.
	SnapshotInterval time.Duration `yaml:"snapshotInterval"`

	// SnapshotOnShutdown triggers an HNSW snapshot during graceful shutdown. Default: true.
	SnapshotOnShutdown bool `yaml:"snapshotOnShutdown"`
}

// GCConfig holds Go runtime garbage collector tuning.
type GCConfig struct {
	// Percent maps to runtime.SetGCPercent(). 100 = default Go behavior.
	// Lower values = more frequent GC = lower peak memory but higher CPU.
	// For a vector DB with large mmap'd memory, 100–200 is typical.
	Percent int `yaml:"percent"`
}

// BloomFilterConfig holds bloom filter tuning for segment-level fast negative lookups.
type BloomFilterConfig struct {
	// Enabled controls whether bloom filters are created during compaction. Default: true
	Enabled bool `yaml:"enabled"`

	// FalsePositiveRate is the target false positive rate (0-1). Default: 0.01 (1%)
	FalsePositiveRate float64 `yaml:"falsePositiveRate"`

	// MinVectors is the minimum vector count before a bloom filter is created.
	// Below this threshold, the overhead isn't justified. Default: 1024
	MinVectors int `yaml:"minVectors"`
}

// WALSyncConfig controls WAL sync behavior for write throughput tuning.
type WALSyncConfig struct {
	// SyncMode controls when WAL writes are fsynced: "per-write", "delayed", or "none".
	// Default: "per-write" (safest, preserves current behavior).
	SyncMode string `yaml:"syncMode"`

	// SyncDelay is the delay before a batched checkpoint in "delayed" mode.
	// Only effective when syncMode is "delayed". Default: "0s" (immediate).
	SyncDelay time.Duration `yaml:"syncDelay"`
}

// ClusterConfig controls distributed cluster behavior.
type ClusterConfig struct {
	// Enabled activates distributed mode. Default: false (single-node).
	Enabled bool `yaml:"enabled"`

	// NodeID is this node's unique identifier in the cluster. Default: "node-1"
	NodeID string `yaml:"nodeID"`

	// ReplicationFactor is the number of copies for each collection. Default: 3
	ReplicationFactor int `yaml:"replicationFactor"`

	// VirtualNodes is the number of virtual nodes per physical node in the hash ring. Default: 150
	VirtualNodes int `yaml:"virtualNodes"`

	// DefaultReadConsistency is the default consistency for search operations. Default: "one"
	DefaultReadConsistency string `yaml:"defaultReadConsistency"`

	// DefaultWriteConsistency is the default consistency for insert operations. Default: "one"
	DefaultWriteConsistency string `yaml:"defaultWriteConsistency"`

	// RequestTimeoutMS is the timeout in milliseconds for inter-node HTTP requests
	// (forwarded inserts, searches, deletes). A value of 0 uses the default of 30s.
	// Large batch inserts or high-dimensional vectors may require higher values.
	RequestTimeoutMS int `yaml:"requestTimeoutMS"`

	// SeedNodes is a list of peer addresses (host:port) to join on startup.
	SeedNodes []string `yaml:"seedNodes"`
}

// GossipConfig controls the gossip protocol for cluster membership.
type GossipConfig struct {
	// Port is the UDP port for gossip messages. Default: 7946
	Port int `yaml:"port"`

	// SecretKey is used for HMAC-SHA256 signing of UDP packets.
	SecretKey string `yaml:"secretKey"`

	// Fanout is the number of random peers to gossip to per round. Default: 3
	Fanout int `yaml:"fanout"`

	// MaxSeen is the maximum number of events to track for dedup. Default: 5000
	MaxSeen int `yaml:"maxSeen"`

	// SeenExpiry is the TTL for seen events. Default: "30s"
	SeenExpiry time.Duration `yaml:"seenExpiry"`

	// ProbeInterval is how often to gossip. Default: "1s"
	ProbeInterval time.Duration `yaml:"probeInterval"`
}

// FailureConfig controls failure detection for cluster nodes.
type FailureConfig struct {
	// Interval is how often to check node health. Default: "1s"
	Interval time.Duration `yaml:"interval"`

	// Timeout is the probe timeout. Default: "500ms"
	Timeout time.Duration `yaml:"timeout"`

	// SuspectAfter is how long before marking suspect. Default: "5s"
	SuspectAfter time.Duration `yaml:"suspectAfter"`

	// DeadAfter is how long before marking dead. Default: "15s"
	DeadAfter time.Duration `yaml:"deadAfter"`
}

// MigrationConfig controls data migration between cluster nodes.
type MigrationConfig struct {
	// BatchSize is vectors per migration batch. Default: 1000
	BatchSize int `yaml:"batchSize"`

	// Parallelism is concurrent migration workers. Default: 4
	Parallelism int `yaml:"parallelism"`

	// MaxRetries is retry count per batch. Default: 3
	MaxRetries int `yaml:"maxRetries"`
}

// DefaultConfig returns a Config populated with all spec-defined defaults.
// Used as the base before overlaying values from config.yaml.
func DefaultConfig() *Config {
	return &Config{
		Server: ServerConfig{
			Addr:              ":8080",
			InternalPort:      ":9090",
			CertFile:          "",
			KeyFile:           "",
			ReadTimeout:       30 * time.Second,
			ReadHeaderTimeout: 5 * time.Second,
			WriteTimeout:      30 * time.Second,
			ShutdownTimeout:   30 * time.Second,
			MaxHeaderBytes:    16384, // 16 KiB
		},
		Auth: AuthConfig{
			KeyHashes: []string{},
		},
		RateLimit: RateLimitConfig{
			SearchRPS:   100,
			SearchBurst: 200,
			InsertRPS:   50,
			InsertBurst: 100,
		},
		Storage: StorageConfig{
			DataDir:      "./data",
			MmapCapacity: 1_000_000,
		},
		Index: IndexConfig{
			Type:               "flat",
			NumClusters:        256,
			DefaultNprobe:      5,
			M:                  16,
			EfConstruction:     200,
			DefaultEf:          50,
			DirtyThreshold:     0.10,
			SnapshotInterval:   5 * time.Minute,
			SnapshotOnShutdown: true,
		},
		GC: GCConfig{
			Percent: 100,
		},
		BloomFilter: BloomFilterConfig{
			Enabled:           true,
			FalsePositiveRate: 0.01,
			MinVectors:        1024,
		},
		WALSync: WALSyncConfig{
			SyncMode:  "per-write",
			SyncDelay: 0,
		},
		Cluster: ClusterConfig{
			Enabled:                 false,
			NodeID:                  "node-1",
			ReplicationFactor:       3,
			VirtualNodes:            150,
			DefaultReadConsistency:  "one",
			DefaultWriteConsistency: "one",
			RequestTimeoutMS:        30000, // 30s default; increase for large batch inserts
			SeedNodes:               []string{},
		},
		Gossip: GossipConfig{
			Port:          7946,
			SecretKey:     "", // empty means no HMAC signing (not recommended for production)
			Fanout:        3,
			MaxSeen:       5000,
			SeenExpiry:    30 * time.Second,
			ProbeInterval: 1 * time.Second,
		},
		Failure: FailureConfig{
			Interval:     1 * time.Second,
			Timeout:      500 * time.Millisecond,
			SuspectAfter: 5 * time.Second,
			DeadAfter:    15 * time.Second,
		},
		Migration: MigrationConfig{
			BatchSize:   1000,
			Parallelism: 4,
			MaxRetries:  3,
		},
	}
}

// LoadConfig reads a YAML config file at path, overlays it on top of DefaultConfig(),
// and validates the result. Returns an error if the file cannot be read or parsed,
// or if validation fails.
func LoadConfig(path string) (*Config, error) {
	cfg := DefaultConfig()

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			// No config file — use built-in defaults. This enables zero-config startup.
			return cfg, nil
		}
		return nil, fmt.Errorf("config: reading %q: %w", path, err)
	}

	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("config: parsing %q: %w", path, err)
	}

	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("config: validation failed: %w", err)
	}

	return cfg, nil
}

// Validate checks that the config values are internally consistent.
// Called automatically by LoadConfig.
func (c *Config) Validate() error {
	if c.Index.M <= 0 {
		return fmt.Errorf("index.m must be > 0, got %d", c.Index.M)
	}
	if c.Index.EfConstruction <= c.Index.M {
		return fmt.Errorf("index.efConstruction (%d) must be > index.m (%d)", c.Index.EfConstruction, c.Index.M)
	}
	if c.Index.DefaultEf <= 0 {
		return fmt.Errorf("index.defaultEf must be > 0, got %d", c.Index.DefaultEf)
	}
	if c.Index.NumClusters <= 0 {
		return fmt.Errorf("index.numClusters must be > 0, got %d", c.Index.NumClusters)
	}
	if c.Index.DirtyThreshold <= 0 || c.Index.DirtyThreshold >= 1.0 {
		return fmt.Errorf("index.dirtyThreshold must be in (0, 1), got %f", c.Index.DirtyThreshold)
	}
	if c.Storage.MmapCapacity <= 0 {
		return fmt.Errorf("storage.mmapCapacity must be > 0, got %d", c.Storage.MmapCapacity)
	}
	validIndexTypes := map[string]bool{"flat": true, "ivf": true, "hnsw": true, "spann": true}
	if !validIndexTypes[c.Index.Type] {
		return fmt.Errorf("index.type must be one of [flat, ivf, hnsw, spann], got %q", c.Index.Type)
	}
	if c.BloomFilter.FalsePositiveRate <= 0 || c.BloomFilter.FalsePositiveRate >= 1.0 {
		return fmt.Errorf("bloomFilter.falsePositiveRate must be in (0, 1), got %f", c.BloomFilter.FalsePositiveRate)
	}
	if c.BloomFilter.MinVectors < 0 {
		return fmt.Errorf("bloomFilter.minVectors must be >= 0, got %d", c.BloomFilter.MinVectors)
	}
	validSyncModes := map[string]bool{"per-write": true, "delayed": true, "none": true}
	if !validSyncModes[c.WALSync.SyncMode] {
		return fmt.Errorf("walSync.syncMode must be one of [per-write, delayed, none], got %q", c.WALSync.SyncMode)
	}
	validConsistency := map[string]bool{"one": true, "quorum": true, "all": true}
	if c.Cluster.DefaultReadConsistency != "" && !validConsistency[c.Cluster.DefaultReadConsistency] {
		return fmt.Errorf("cluster.defaultReadConsistency must be one of [one, quorum, all], got %q", c.Cluster.DefaultReadConsistency)
	}
	if c.Cluster.DefaultWriteConsistency != "" && !validConsistency[c.Cluster.DefaultWriteConsistency] {
		return fmt.Errorf("cluster.defaultWriteConsistency must be one of [one, quorum, all], got %q", c.Cluster.DefaultWriteConsistency)
	}
	if c.Gossip.Port < 0 {
		return fmt.Errorf("gossip.port must be >= 0, got %d", c.Gossip.Port)
	}
	if c.Gossip.Fanout < 0 {
		return fmt.Errorf("gossip.fanout must be >= 0, got %d", c.Gossip.Fanout)
	}
	if c.Migration.BatchSize < 0 {
		return fmt.Errorf("migration.batchSize must be >= 0, got %d", c.Migration.BatchSize)
	}
	return nil
}
