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
	Server    ServerConfig    `yaml:"server"`
	Auth      AuthConfig      `yaml:"auth"`
	RateLimit RateLimitConfig `yaml:"rateLimit"`
	Storage   StorageConfig   `yaml:"storage"`
	Index     IndexConfig     `yaml:"index"`
	GC        GCConfig        `yaml:"gc"`
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
}

// GCConfig holds Go runtime garbage collector tuning.
type GCConfig struct {
	// Percent maps to runtime.SetGCPercent(). 100 = default Go behavior.
	// Lower values = more frequent GC = lower peak memory but higher CPU.
	// For a vector DB with large mmap'd memory, 100–200 is typical.
	Percent int `yaml:"percent"`
}

// DefaultConfig returns a Config populated with all spec-defined defaults.
// Used as the base before overlaying values from config.yaml.
func DefaultConfig() *Config {
	return &Config{
		Server: ServerConfig{
			Addr:              ":8443",
			InternalPort:      ":9090",
			CertFile:          "certs/server.crt",
			KeyFile:           "certs/server.key",
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
			Type:           "flat",
			NumClusters:    256,
			DefaultNprobe:  5,
			M:              16,
			EfConstruction: 200,
			DefaultEf:      50,
			DirtyThreshold: 0.10,
		},
		GC: GCConfig{
			Percent: 100,
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
	return nil
}
