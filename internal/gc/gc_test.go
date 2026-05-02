package gc

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ── PinTracker Tests ────────────────────────────────────────────────────────

func TestPinTracker_PinUnpin(t *testing.T) {
	pt := NewPinTracker()

	pt.Pin("seg-1")
	assert.Equal(t, int64(1), pt.PinCount("seg-1"))

	pt.Pin("seg-1")
	assert.Equal(t, int64(2), pt.PinCount("seg-1"))

	pt.Unpin("seg-1")
	assert.Equal(t, int64(1), pt.PinCount("seg-1"))

	pt.Unpin("seg-1")
	assert.Equal(t, int64(0), pt.PinCount("seg-1"))
}

func TestPinTracker_UnpinNonExistent(t *testing.T) {
	pt := NewPinTracker()
	pt.Unpin("nope") // should not panic
	assert.Equal(t, int64(0), pt.PinCount("nope"))
}

func TestPinTracker_Remove(t *testing.T) {
	pt := NewPinTracker()
	pt.Pin("seg-1")
	pt.Remove("seg-1")
	assert.Equal(t, int64(0), pt.PinCount("seg-1"))
}

// ── Collector Tests ─────────────────────────────────────────────────────────

func TestCollector_MarkAndSweep(t *testing.T) {
	dir := t.TempDir()

	// Create a temp file to be collected
	fpath := filepath.Join(dir, "old_segment.bin")
	require.NoError(t, os.WriteFile(fpath, []byte("segment data"), 0600))

	pt := NewPinTracker()
	cfg := GCConfig{
		Interval:     time.Second,
		FenceTimeout: time.Second,
		MinAge:       0, // no minimum age for test
	}
	gc := NewCollector(pt, cfg)

	// Mark the segment
	gc.Mark(SegmentRef{
		ID:       "seg-old",
		FilePath: fpath,
		MarkedAt: time.Now().Add(-time.Minute), // already old enough
	})
	assert.Equal(t, 1, gc.MarkedCount())

	// Sweep
	ctx := context.Background()
	collected, err := gc.Sweep(ctx)
	require.NoError(t, err)
	assert.Equal(t, 1, collected)
	assert.Equal(t, 0, gc.MarkedCount())

	// File should be deleted
	_, err = os.Stat(fpath)
	assert.True(t, os.IsNotExist(err), "segment file should be deleted")
}

func TestCollector_FenceBlocksSweep_Pinned(t *testing.T) {
	dir := t.TempDir()
	fpath := filepath.Join(dir, "pinned_segment.bin")
	require.NoError(t, os.WriteFile(fpath, []byte("data"), 0600))

	pt := NewPinTracker()
	pt.Pin("seg-pinned") // active reader

	cfg := GCConfig{
		Interval:     time.Second,
		FenceTimeout: time.Second,
		MinAge:       0,
	}
	gc := NewCollector(pt, cfg)

	gc.Mark(SegmentRef{
		ID:       "seg-pinned",
		FilePath: fpath,
		MarkedAt: time.Now().Add(-time.Minute),
	})

	// Sweep should NOT collect the pinned segment
	collected, err := gc.Sweep(context.Background())
	require.NoError(t, err)
	assert.Equal(t, 0, collected)
	assert.Equal(t, 1, gc.MarkedCount(), "pinned segment should remain in queue")

	// File should still exist
	_, err = os.Stat(fpath)
	assert.NoError(t, err, "pinned segment file should not be deleted")

	// Unpin and sweep again
	pt.Unpin("seg-pinned")
	collected, err = gc.Sweep(context.Background())
	require.NoError(t, err)
	assert.Equal(t, 1, collected)
}

func TestCollector_FenceBlocksSweep_MinAge(t *testing.T) {
	dir := t.TempDir()
	fpath := filepath.Join(dir, "young_segment.bin")
	require.NoError(t, os.WriteFile(fpath, []byte("data"), 0600))

	pt := NewPinTracker()
	cfg := GCConfig{
		Interval:     time.Second,
		FenceTimeout: time.Second,
		MinAge:       10 * time.Minute, // 10 minutes minimum
	}
	gc := NewCollector(pt, cfg)

	gc.Mark(SegmentRef{
		ID:       "seg-young",
		FilePath: fpath,
		MarkedAt: time.Now(), // just marked — too young
	})

	collected, err := gc.Sweep(context.Background())
	require.NoError(t, err)
	assert.Equal(t, 0, collected, "young segment should not be collected")
	assert.Equal(t, 1, gc.MarkedCount())
}

func TestCollector_SweepMissingFile(t *testing.T) {
	pt := NewPinTracker()
	cfg := GCConfig{MinAge: 0}
	gc := NewCollector(pt, cfg)

	gc.Mark(SegmentRef{
		ID:       "seg-gone",
		FilePath: "/nonexistent/path/segment.bin",
		MarkedAt: time.Now().Add(-time.Hour),
	})

	// Sweep should succeed (os.Remove returns not-exist, which we treat as success)
	collected, err := gc.Sweep(context.Background())
	require.NoError(t, err)
	assert.Equal(t, 1, collected)
}

func TestCollector_SweepNoFilePath(t *testing.T) {
	pt := NewPinTracker()
	cfg := GCConfig{MinAge: 0}
	gc := NewCollector(pt, cfg)

	gc.Mark(SegmentRef{
		ID:       "seg-meta-only",
		FilePath: "", // metadata-only segment (e.g., S3 key deleted externally)
		MarkedAt: time.Now().Add(-time.Hour),
	})

	collected, err := gc.Sweep(context.Background())
	require.NoError(t, err)
	assert.Equal(t, 1, collected)
}

func TestCollector_Stats(t *testing.T) {
	pt := NewPinTracker()
	cfg := GCConfig{MinAge: 0}
	gc := NewCollector(pt, cfg)

	gc.Mark(SegmentRef{ID: "a", MarkedAt: time.Now().Add(-time.Hour)})
	gc.Mark(SegmentRef{ID: "b", MarkedAt: time.Now().Add(-time.Hour)})

	stats := gc.Stats()
	assert.Equal(t, 2, stats.PendingSegments)
	assert.Equal(t, int64(0), stats.TotalCollected)

	gc.Sweep(context.Background())

	stats = gc.Stats()
	assert.Equal(t, 0, stats.PendingSegments)
	assert.Equal(t, int64(2), stats.TotalCollected)
}

func TestCollector_SweepCancellation(t *testing.T) {
	pt := NewPinTracker()
	cfg := GCConfig{MinAge: 0}
	gc := NewCollector(pt, cfg)

	for i := 0; i < 100; i++ {
		gc.Mark(SegmentRef{
			ID:       "seg-" + string(rune('a'+i%26)),
			MarkedAt: time.Now().Add(-time.Hour),
		})
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	_, err := gc.Sweep(ctx)
	assert.ErrorIs(t, err, context.Canceled)
}

func TestCollector_Run(t *testing.T) {
	pt := NewPinTracker()
	cfg := GCConfig{
		Interval: 50 * time.Millisecond,
		MinAge:   0,
	}
	gc := NewCollector(pt, cfg)

	gc.Mark(SegmentRef{ID: "seg-run", MarkedAt: time.Now().Add(-time.Hour)})

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	go gc.Run(ctx)

	// Wait for the GC to run at least once
	time.Sleep(150 * time.Millisecond)

	stats := gc.Stats()
	assert.Equal(t, int64(1), stats.TotalCollected)
}

func TestCollector_MultipleSegments(t *testing.T) {
	dir := t.TempDir()
	pt := NewPinTracker()
	cfg := GCConfig{MinAge: 0}
	gc := NewCollector(pt, cfg)

	// Create 5 segment files
	for i := 0; i < 5; i++ {
		fpath := filepath.Join(dir, "seg_"+string(rune('a'+i))+".bin")
		require.NoError(t, os.WriteFile(fpath, []byte("data"), 0600))
		gc.Mark(SegmentRef{
			ID:       "seg-" + string(rune('a'+i)),
			FilePath: fpath,
			MarkedAt: time.Now().Add(-time.Hour),
		})
	}

	// Pin one segment
	pt.Pin("seg-c")

	collected, err := gc.Sweep(context.Background())
	require.NoError(t, err)
	assert.Equal(t, 4, collected, "4 of 5 should be collected")
	assert.Equal(t, 1, gc.MarkedCount(), "1 pinned segment remains")
}
