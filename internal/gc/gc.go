// Package gc implements three-phase garbage collection for vector index segments.
//
// Architecture (from ChromaDB's rust/garbage_collector):
//
//	Phase 1 — MARK: Logical deletion written to WAL. Item invisible
//	  to queries via tombstone masking. Old segment files still on disk.
//
//	Phase 2 — FENCE: Wait for all in-flight readers to release their
//	  segment pins. A pin is an atomic counter incremented when a goroutine
//	  begins reading a segment version and decremented when it finishes.
//	  When pin count reaches zero, the segment is safe to collect.
//
//	Phase 3 — SWEEP: Physically delete old segment files and purge
//	  tombstones from the WAL/metadata. Update SysDB to remove references.
//
// This design is identical to epoch-based reclamation in lock-free data
// structures: the fence phase prevents use-after-free on segment files.
package gc

import (
	"context"
	"log/slog"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

// SegmentRef represents a tracked segment file eligible for collection.
type SegmentRef struct {
	ID       string
	FilePath string
	MarkedAt time.Time
}

// PinTracker tracks active readers for segment files.
// A segment cannot be swept while it has active pins.
type PinTracker struct {
	mu   sync.RWMutex
	pins map[string]*atomic.Int64 // segment_id → active reader count
}

// NewPinTracker creates a new pin tracker.
func NewPinTracker() *PinTracker {
	return &PinTracker{
		pins: make(map[string]*atomic.Int64),
	}
}

// Pin increments the reader count for a segment.
// Call this before starting to read a segment file.
func (pt *PinTracker) Pin(segmentID string) {
	pt.mu.Lock()
	counter, ok := pt.pins[segmentID]
	if !ok {
		counter = &atomic.Int64{}
		pt.pins[segmentID] = counter
	}
	pt.mu.Unlock()
	counter.Add(1)
}

// Unpin decrements the reader count for a segment.
// Call this when done reading a segment file (deferred).
func (pt *PinTracker) Unpin(segmentID string) {
	pt.mu.RLock()
	counter, ok := pt.pins[segmentID]
	pt.mu.RUnlock()
	if ok {
		counter.Add(-1)
	}
}

// PinCount returns the current pin count for a segment.
func (pt *PinTracker) PinCount(segmentID string) int64 {
	pt.mu.RLock()
	counter, ok := pt.pins[segmentID]
	pt.mu.RUnlock()
	if !ok {
		return 0
	}
	return counter.Load()
}

// Remove removes a segment from the pin tracker entirely.
func (pt *PinTracker) Remove(segmentID string) {
	pt.mu.Lock()
	delete(pt.pins, segmentID)
	pt.mu.Unlock()
}

// GCConfig configures the garbage collector.
type GCConfig struct {
	Interval      time.Duration // how often to run sweeps (default: 60s)
	FenceTimeout  time.Duration // max time to wait for readers to drain (default: 30s)
	MinAge        time.Duration // minimum age before a marked segment can be swept (default: 5m)
}

// DefaultGCConfig returns production-ready defaults.
func DefaultGCConfig() GCConfig {
	return GCConfig{
		Interval:     60 * time.Second,
		FenceTimeout: 30 * time.Second,
		MinAge:       5 * time.Minute,
	}
}

// Collector implements three-phase garbage collection.
type Collector struct {
	mu       sync.Mutex
	marked   []SegmentRef // Phase 1: segments waiting for collection
	pins     *PinTracker
	config   GCConfig
	logger   *slog.Logger

	// Stats
	totalCollected atomic.Int64
	totalBytes     atomic.Int64
}

// NewCollector creates a garbage collector.
func NewCollector(pins *PinTracker, config GCConfig) *Collector {
	return &Collector{
		pins:   pins,
		config: config,
		logger: slog.Default().With("component", "gc"),
	}
}

// Mark registers a segment for garbage collection (Phase 1).
// The segment is not deleted immediately — it enters the fence queue.
func (gc *Collector) Mark(seg SegmentRef) {
	gc.mu.Lock()
	defer gc.mu.Unlock()
	if seg.MarkedAt.IsZero() {
		seg.MarkedAt = time.Now()
	}
	gc.marked = append(gc.marked, seg)
	gc.logger.Info("segment marked for GC",
		"segment_id", seg.ID,
		"file", seg.FilePath,
	)
}

// MarkedCount returns the number of segments waiting for collection.
func (gc *Collector) MarkedCount() int {
	gc.mu.Lock()
	defer gc.mu.Unlock()
	return len(gc.marked)
}

// Sweep performs one garbage collection pass (Phases 2 + 3).
// Returns the number of segments collected and any error.
func (gc *Collector) Sweep(ctx context.Context) (int, error) {
	gc.mu.Lock()
	pending := make([]SegmentRef, len(gc.marked))
	copy(pending, gc.marked)
	gc.mu.Unlock()

	now := time.Now()
	var collected int
	var remaining []SegmentRef

	for _, seg := range pending {
		select {
		case <-ctx.Done():
			// Put remaining back
			gc.mu.Lock()
			gc.marked = append(remaining, pending[collected:]...)
			gc.mu.Unlock()
			return collected, ctx.Err()
		default:
		}

		// Phase 2: FENCE — Check minimum age
		age := now.Sub(seg.MarkedAt)
		if age < gc.config.MinAge {
			remaining = append(remaining, seg)
			continue
		}

		// Phase 2: FENCE — Check pin count
		pins := gc.pins.PinCount(seg.ID)
		if pins > 0 {
			gc.logger.Debug("segment still pinned, skipping",
				"segment_id", seg.ID,
				"pins", pins,
			)
			remaining = append(remaining, seg)
			continue
		}

		// Phase 3: SWEEP — Delete the file
		if seg.FilePath != "" {
			info, _ := os.Stat(seg.FilePath)
			var fileSize int64
			if info != nil {
				fileSize = info.Size()
			}

			if err := os.Remove(seg.FilePath); err != nil && !os.IsNotExist(err) {
				gc.logger.Error("failed to delete segment file",
					"segment_id", seg.ID,
					"file", seg.FilePath,
					"error", err,
				)
				remaining = append(remaining, seg)
				continue
			}

			gc.totalBytes.Add(fileSize)
		}

		// Clean up pin tracker
		gc.pins.Remove(seg.ID)
		gc.totalCollected.Add(1)
		collected++

		gc.logger.Info("segment collected",
			"segment_id", seg.ID,
			"file", seg.FilePath,
			"age", age.Round(time.Second),
		)
	}

	// Update marked list with remaining segments
	gc.mu.Lock()
	gc.marked = remaining
	gc.mu.Unlock()

	return collected, nil
}

// Run starts the garbage collector loop. Blocks until ctx is cancelled.
func (gc *Collector) Run(ctx context.Context) {
	ticker := time.NewTicker(gc.config.Interval)
	defer ticker.Stop()

	gc.logger.Info("garbage collector started",
		"interval", gc.config.Interval,
		"min_age", gc.config.MinAge,
	)

	for {
		select {
		case <-ctx.Done():
			gc.logger.Info("garbage collector stopped")
			return
		case <-ticker.C:
			n, err := gc.Sweep(ctx)
			if err != nil && ctx.Err() == nil {
				gc.logger.Error("GC sweep failed", "error", err)
			}
			if n > 0 {
				gc.logger.Info("GC sweep completed",
					"collected", n,
					"total_collected", gc.totalCollected.Load(),
					"total_bytes_freed", gc.totalBytes.Load(),
				)
			}
		}
	}
}

// Stats returns garbage collector statistics.
func (gc *Collector) Stats() GCStats {
	gc.mu.Lock()
	pending := len(gc.marked)
	gc.mu.Unlock()
	return GCStats{
		PendingSegments: pending,
		TotalCollected:  gc.totalCollected.Load(),
		TotalBytesFreed: gc.totalBytes.Load(),
	}
}

// GCStats holds garbage collector performance counters.
type GCStats struct {
	PendingSegments int
	TotalCollected  int64
	TotalBytesFreed int64
}
