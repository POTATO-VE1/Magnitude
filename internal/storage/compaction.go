// Package storage — compaction worker.
// The Compactor reads WAL entries and materializes them into indexed segments
// stored on disk. After compaction, the corresponding WAL entries are truncated.
//
// ChromaDB Architecture: The Compactor is one of the five core components.
// It decouples write durability (WAL) from read performance (indexed segments).
//
// ChromaDB Lesson 9: "Blocks and Blockfiles, not raw files."
// Compacted segments are immutable after write. Mutation = new file + atomic rename.
package storage

import (
	"context"
	"crypto/sha256"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"time"
	"unsafe"

	"github.com/POTATO-VE1/Magnitude/internal/events"
)

// Compactor materializes vectors from an in-memory index to disk.
// Compaction produces an immutable segment file with:
//   - A validated FileHeader (magic, version, checksum)
//   - Dense float32 vector data in row-major layout
//
// After compaction, the corresponding WAL entries are truncated.
type Compactor struct {
	mu        sync.Mutex
	dataDir   string
	interval  time.Duration
	cancel    context.CancelFunc
	done      chan struct{}
	running   bool
	flowBus   *events.FlowBus           // optional
	onCompact func(collectionID string) // called after successful compaction per collection
}

// CompactorOption configures the Compactor.
type CompactorOption func(*Compactor)

// WithCompactorFlowBus sets the event bus for the Compactor.
func WithCompactorFlowBus(bus *events.FlowBus) CompactorOption {
	return func(c *Compactor) { c.flowBus = bus }
}

// WithCompactionCallback sets a function called after successful compaction
// for each affected collection. Used to trigger HNSW snapshots.
func WithCompactionCallback(fn func(collectionID string)) CompactorOption {
	return func(c *Compactor) { c.onCompact = fn }
}

// NewCompactor creates a new compaction worker.
func NewCompactor(dataDir string, interval time.Duration, opts ...CompactorOption) *Compactor {
	c := &Compactor{
		dataDir:  dataDir,
		interval: interval,
		done:     make(chan struct{}),
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Start begins the background compaction loop. Idempotent — calling Start
// on an already-running compactor is a no-op.
func (c *Compactor) Start(compactFn func() error) {
	c.mu.Lock()
	if c.running {
		c.mu.Unlock()
		return
	}
	c.running = true
	c.mu.Unlock()

	ctx, cancel := context.WithCancel(context.Background())
	c.mu.Lock()
	c.cancel = cancel
	c.done = make(chan struct{})
	c.mu.Unlock()

	go func() {
		defer func() {
			c.mu.Lock()
			c.running = false
			c.mu.Unlock()
			close(c.done)
		}()
		ticker := time.NewTicker(c.interval)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				if err := compactFn(); err != nil {
					slog.Error("compaction failed", "error", err)
				} else if c.flowBus != nil {
					c.flowBus.Notify(events.EventCompactionComplete)
				}
			}
		}
	}()
}

// Stop halts the background compaction loop and waits for it to finish.
func (c *Compactor) Stop() {
	c.mu.Lock()
	cancel := c.cancel
	done := c.done
	c.mu.Unlock()

	if cancel != nil {
		cancel()
		<-done
	}
}

// CompactVectors writes vectors to a new segment file with an atomic rename.
// This ensures crash safety: readers either see the old file or the new file,
// never a partially-written intermediate.
//
// Process:
//  1. Write to a temp file in the same directory
//  2. Compute SHA-256 checksum of the data section
//  3. Write the header with the checksum
//  4. Flush (fsync) the temp file
//  5. Atomic rename to the target path
func CompactVectors(targetPath string, vectors []float32, dim int) error {
	if dim <= 0 {
		return fmt.Errorf("compaction: invalid dimension %d", dim)
	}
	vectorCount := len(vectors) / dim
	if vectorCount == 0 {
		return nil // nothing to compact
	}

	// Write to temp file in the same directory for atomic rename
	dir := filepath.Dir(targetPath)
	tmpFile, err := os.CreateTemp(dir, "compact-*.tmp")
	if err != nil {
		return fmt.Errorf("compaction: creating temp file: %w", err)
	}
	tmpPath := tmpFile.Name()
	defer func() {
		// Clean up on failure
		if tmpFile != nil {
			tmpFile.Close()
			os.Remove(tmpPath)
		}
	}()

	// Write placeholder header (will overwrite with checksum later)
	headerBuf := make([]byte, headerSize)
	if _, err := tmpFile.Write(headerBuf); err != nil {
		return fmt.Errorf("compaction: writing header placeholder: %w", err)
	}

	// Write vector data as raw float32 bytes (zero-copy via unsafe.Slice)
	dataLen := len(vectors) * 4
	dataBuf := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(vectors))), dataLen)
	if _, err := tmpFile.Write(dataBuf); err != nil {
		return fmt.Errorf("compaction: writing vector data: %w", err)
	}

	// Compute checksum of the data section
	checksum := sha256.Sum256(dataBuf)

	// Seek back to beginning and write the real header
	if _, err := tmpFile.Seek(0, 0); err != nil {
		return fmt.Errorf("compaction: seeking to header: %w", err)
	}
	if err := WriteHeader(tmpFile, uint64(vectorCount), uint32(dim), checksum); err != nil {
		return fmt.Errorf("compaction: writing header: %w", err)
	}

	// Fsync to ensure durability before rename
	if err := tmpFile.Sync(); err != nil {
		return fmt.Errorf("compaction: fsync: %w", err)
	}
	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("compaction: closing temp file: %w", err)
	}
	tmpFile = nil // prevent deferred cleanup

	// Atomic rename: crash-safe swap
	if err := os.Rename(tmpPath, targetPath); err != nil {
		return fmt.Errorf("compaction: atomic rename %q → %q: %w", tmpPath, targetPath, err)
	}

	// Fsync directory to ensure rename is durably recorded
	if dirFile, err := os.Open(dir); err == nil {
		_ = dirFile.Sync()
		_ = dirFile.Close()
	} else {
		slog.Warn("compaction: failed to open directory for fsync", "dir", dir, "error", err)
	}

	slog.Info("compaction complete",
		"path", targetPath,
		"vectors", vectorCount,
		"dim", dim,
		"bytes", len(dataBuf)+headerSize,
	)

	return nil
}
