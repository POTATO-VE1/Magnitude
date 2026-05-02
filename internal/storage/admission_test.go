package storage

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ── Admission Controller Tests ──────────────────────────────────────────────

func TestAdmissionController_BasicAcquireRelease(t *testing.T) {
	ac := NewAdmissionController(5)

	for i := 0; i < 5; i++ {
		err := ac.Acquire(context.Background(), "prefix-a")
		require.NoError(t, err)
	}

	assert.Equal(t, 5, ac.InFlight("prefix-a"))

	for i := 0; i < 5; i++ {
		ac.Release("prefix-a")
	}
	assert.Equal(t, 0, ac.InFlight("prefix-a"))
}

func TestAdmissionController_BlocksAtLimit(t *testing.T) {
	ac := NewAdmissionController(2)

	// Fill both slots
	require.NoError(t, ac.Acquire(context.Background(), "p"))
	require.NoError(t, ac.Acquire(context.Background(), "p"))

	// Third acquire should block
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	err := ac.Acquire(ctx, "p")
	assert.ErrorIs(t, err, context.DeadlineExceeded)

	stats := ac.Stats()
	assert.Equal(t, int64(3), stats.TotalAcquires)
	assert.Equal(t, int64(1), stats.TotalWaits)
	assert.Equal(t, int64(1), stats.TotalTimeouts)
}

func TestAdmissionController_WakerRelease(t *testing.T) {
	ac := NewAdmissionController(1)

	// Take the slot
	require.NoError(t, ac.Acquire(context.Background(), "p"))

	// Waiter goroutine
	done := make(chan error, 1)
	go func() {
		done <- ac.Acquire(context.Background(), "p")
	}()

	// Give the goroutine time to queue
	time.Sleep(20 * time.Millisecond)
	assert.Equal(t, 1, ac.QueueDepth("p"))

	// Release should wake the waiter
	ac.Release("p")

	err := <-done
	assert.NoError(t, err)
}

func TestAdmissionController_MultiplePrefixes(t *testing.T) {
	ac := NewAdmissionController(2)

	// Prefix A: fill 2 slots
	require.NoError(t, ac.Acquire(context.Background(), "a"))
	require.NoError(t, ac.Acquire(context.Background(), "a"))

	// Prefix B: independent, should succeed
	require.NoError(t, ac.Acquire(context.Background(), "b"))

	assert.Equal(t, 2, ac.InFlight("a"))
	assert.Equal(t, 1, ac.InFlight("b"))
}

func TestAdmissionController_Concurrent(t *testing.T) {
	ac := NewAdmissionController(10)
	var wg sync.WaitGroup

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			err := ac.Acquire(context.Background(), "concurrent")
			if err != nil {
				return
			}
			time.Sleep(time.Millisecond)
			ac.Release("concurrent")
		}()
	}
	wg.Wait()

	assert.Equal(t, 0, ac.InFlight("concurrent"))
	stats := ac.Stats()
	assert.Equal(t, int64(100), stats.TotalAcquires)
}

// ── KeyPrefix Tests ─────────────────────────────────────────────────────────

func TestKeyPrefix(t *testing.T) {
	tests := []struct {
		key    string
		prefix string
	}{
		{"logs/tenant1/db1/collection1/frag.parquet", "logs/tenant1"},
		{"segments/tenant1/db1/collection1/hnsw/data.bin", "segments/tenant1"},
		{"a/b", "a/b"},
		{"singlekey", "singlekey"},
	}

	for _, tt := range tests {
		t.Run(tt.key, func(t *testing.T) {
			assert.Equal(t, tt.prefix, KeyPrefix(tt.key))
		})
	}
}

// ── Memory Store Tests ──────────────────────────────────────────────────────

func TestMemoryStore_CRUD(t *testing.T) {
	store := NewMemoryStore(0)
	ctx := context.Background()

	// Put
	require.NoError(t, store.Put(ctx, "bucket", "key1", []byte("hello")))

	// Get
	data, err := store.Get(ctx, "bucket", "key1")
	require.NoError(t, err)
	assert.Equal(t, []byte("hello"), data)

	// Delete
	require.NoError(t, store.Delete(ctx, "bucket", "key1"))
	_, err = store.Get(ctx, "bucket", "key1")
	assert.Error(t, err)
}

func TestMemoryStore_ConditionalPut(t *testing.T) {
	store := NewMemoryStore(0)
	ctx := context.Background()

	// First write (no existing ETag)
	etag1, err := store.ConditionalPut(ctx, "b", "manifest", []byte("v1"), "")
	require.NoError(t, err)
	assert.NotEmpty(t, etag1)

	// Second write with correct ETag
	etag2, err := store.ConditionalPut(ctx, "b", "manifest", []byte("v2"), etag1)
	require.NoError(t, err)
	assert.NotEqual(t, etag1, etag2)

	// Third write with stale ETag should fail
	_, err = store.ConditionalPut(ctx, "b", "manifest", []byte("v3"), etag1)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "precondition failed")
}

func TestAdmittedStore_Integration(t *testing.T) {
	inner := NewMemoryStore(0)
	store := NewAdmittedStore(inner, 5)
	ctx := context.Background()

	// Normal operations should work
	require.NoError(t, store.Put(ctx, "bucket", "k1", []byte("data")))
	data, err := store.Get(ctx, "bucket", "k1")
	require.NoError(t, err)
	assert.Equal(t, []byte("data"), data)

	require.NoError(t, store.Delete(ctx, "bucket", "k1"))
	_, err = store.Get(ctx, "bucket", "k1")
	assert.Error(t, err)

	stats := store.AdmissionStats()
	assert.Equal(t, int64(4), stats.TotalAcquires) // Put + Get + Delete + failed Get
}

func TestAdmittedStore_ConcurrentWithLimit(t *testing.T) {
	inner := NewMemoryStore(5 * time.Millisecond) // simulate S3 latency
	store := NewAdmittedStore(inner, 3)            // max 3 concurrent per prefix
	ctx := context.Background()

	// Pre-populate some data
	for i := 0; i < 10; i++ {
		store.Put(ctx, "bucket", "logs/tenant1/data"+string(rune('0'+i)), []byte("data"))
	}

	// Concurrent reads from the same prefix
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			key := "logs/tenant1/data" + string(rune('0'+i%10))
			store.Get(ctx, "bucket", key)
		}(i)
	}
	wg.Wait()

	stats := store.AdmissionStats()
	t.Logf("Admitted store: %d acquires, %d waits, %d timeouts",
		stats.TotalAcquires, stats.TotalWaits, stats.TotalTimeouts)
}
