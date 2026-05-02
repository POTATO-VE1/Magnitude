package cache

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSegmentCache_PutAndGet(t *testing.T) {
	c := NewSegmentCache(1024)

	// First access: should NOT be cached (admission control)
	admitted := c.Put("seg-1", []byte("hello"))
	assert.False(t, admitted, "first access should not admit to cache")
	assert.Nil(t, c.Get("seg-1"), "should not be in cache after first access")

	// Second access: should be cached
	admitted = c.Put("seg-1", []byte("hello"))
	assert.True(t, admitted, "second access should admit to cache")
	assert.Equal(t, []byte("hello"), c.Get("seg-1"))
}

func TestSegmentCache_Get_Miss(t *testing.T) {
	c := NewSegmentCache(1024)
	assert.Nil(t, c.Get("nonexistent"))
}

func TestSegmentCache_LRUEviction(t *testing.T) {
	// Cache with 20 bytes capacity
	c := NewSegmentCache(20)

	// Admit "a" (10 bytes)
	c.Put("a", make([]byte, 10))
	c.Put("a", make([]byte, 10))
	assert.Equal(t, 1, c.Len())

	// Admit "b" (10 bytes) — fills cache
	c.Put("b", make([]byte, 10))
	c.Put("b", make([]byte, 10))
	assert.Equal(t, 2, c.Len())

	// Admit "c" (10 bytes) — should evict "a" (LRU)
	c.Put("c", make([]byte, 10))
	c.Put("c", make([]byte, 10))

	assert.Nil(t, c.Get("a"), "a should be evicted (LRU)")
	assert.NotNil(t, c.Get("b"), "b should still be in cache")
	assert.NotNil(t, c.Get("c"), "c should be in cache")
}

func TestSegmentCache_LRU_AccessOrder(t *testing.T) {
	c := NewSegmentCache(30)

	// Admit a, b, c (10 bytes each)
	for _, key := range []string{"a", "b", "c"} {
		c.Put(key, make([]byte, 10))
		c.Put(key, make([]byte, 10))
	}
	assert.Equal(t, 3, c.Len())

	// Access "a" to move it to front
	c.Get("a")

	// Insert "d" — should evict "b" (now LRU, since "a" was recently accessed)
	c.Put("d", make([]byte, 10))
	c.Put("d", make([]byte, 10))

	assert.NotNil(t, c.Get("a"), "a should survive (recently accessed)")
	assert.Nil(t, c.Get("b"), "b should be evicted (LRU)")
}

func TestSegmentCache_Update(t *testing.T) {
	c := NewSegmentCache(1024)

	c.Put("seg-1", []byte("v1"))
	c.Put("seg-1", []byte("v1")) // admit
	assert.Equal(t, []byte("v1"), c.Get("seg-1"))

	// Update value
	c.Put("seg-1", []byte("v2-updated"))
	assert.Equal(t, []byte("v2-updated"), c.Get("seg-1"))
}

func TestSegmentCache_Remove(t *testing.T) {
	c := NewSegmentCache(1024)

	c.Put("seg-1", []byte("data"))
	c.Put("seg-1", []byte("data")) // admit
	require.NotNil(t, c.Get("seg-1"))

	c.Remove("seg-1")
	assert.Nil(t, c.Get("seg-1"))
	assert.Equal(t, 0, c.Len())

	// Re-insert requires going through admission again
	admitted := c.Put("seg-1", []byte("data"))
	assert.False(t, admitted, "after Remove, seen filter should be cleared for this key")
}

func TestSegmentCache_Remove_NonExistent(t *testing.T) {
	c := NewSegmentCache(1024)
	c.Remove("nope") // should not panic
}

func TestSegmentCache_OversizedEntry(t *testing.T) {
	c := NewSegmentCache(10)

	// Entry larger than cache capacity
	c.Put("huge", make([]byte, 100))
	admitted := c.Put("huge", make([]byte, 100))
	assert.False(t, admitted, "entry larger than cache should not be admitted")
	assert.Equal(t, 0, c.Len())
}

func TestSegmentCache_Clear(t *testing.T) {
	c := NewSegmentCache(1024)

	c.Put("a", []byte("1"))
	c.Put("a", []byte("1"))
	c.Put("b", []byte("2"))
	c.Put("b", []byte("2"))

	c.Clear()
	assert.Equal(t, 0, c.Len())
	assert.Equal(t, int64(0), c.SizeBytes())
	assert.Nil(t, c.Get("a"))
}

func TestSegmentCache_Stats(t *testing.T) {
	c := NewSegmentCache(1024)

	c.Put("a", make([]byte, 100))
	c.Put("a", make([]byte, 100))

	stats := c.Stats()
	assert.Equal(t, 1, stats.Entries)
	assert.Equal(t, int64(100), stats.CurrentBytes)
	assert.Equal(t, int64(1024), stats.MaxBytes)
	assert.GreaterOrEqual(t, stats.SeenKeys, 1)
}

func TestSegmentCache_SizeTracking(t *testing.T) {
	c := NewSegmentCache(1024)

	c.Put("a", make([]byte, 50))
	c.Put("a", make([]byte, 50))
	assert.Equal(t, int64(50), c.SizeBytes())

	c.Put("b", make([]byte, 30))
	c.Put("b", make([]byte, 30))
	assert.Equal(t, int64(80), c.SizeBytes())

	c.Remove("a")
	assert.Equal(t, int64(30), c.SizeBytes())
}

func TestSegmentCache_Concurrent(t *testing.T) {
	c := NewSegmentCache(10000)
	var wg sync.WaitGroup

	// 50 writers + 50 readers
	for i := 0; i < 50; i++ {
		wg.Add(2)
		go func(n int) {
			defer wg.Done()
			key := fmt.Sprintf("key-%d", n)
			c.Put(key, make([]byte, 10))
			c.Put(key, make([]byte, 10))
		}(i)
		go func(n int) {
			defer wg.Done()
			key := fmt.Sprintf("key-%d", n)
			c.Get(key) // may or may not be present
		}(i)
	}
	wg.Wait()

	assert.LessOrEqual(t, c.SizeBytes(), int64(10000))
}

func BenchmarkSegmentCache_Put(b *testing.B) {
	c := NewSegmentCache(1 << 20) // 1MB
	data := make([]byte, 100)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		key := fmt.Sprintf("key-%d", i)
		c.Put(key, data)
		c.Put(key, data)
	}
}
