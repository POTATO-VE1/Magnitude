// Package cache implements an LRU segment cache with admission control.
//
// Architecture:
//
//	Query → SegmentCache.GetOrLoad(key) →
//	  HIT:  return cached data, move to front of LRU
//	  MISS: load from backend, check admission filter
//	         ├─ FIRST ACCESS: add to bloom filter, return without caching
//	         └─ SECOND ACCESS: admit to LRU cache
//
// The second-access admission policy prevents one-time bulk scans (compaction,
// full-index rebuild) from evicting frequently accessed query segments.
//
// ChromaDB uses this pattern for SSD-cached S3 segments.
package cache

import (
	"container/list"
	"sync"
)

// entry represents a cached segment.
type entry struct {
	key   string
	value []byte
	size  int64
	elem  *list.Element
}

// SegmentCache is an LRU cache with a second-access admission policy.
type SegmentCache struct {
	mu           sync.Mutex
	entries      map[string]*entry // key → cached entry
	lru          *list.List        // LRU ordering (front = most recent)
	maxBytes     int64
	currentBytes int64
	seenFilter   map[string]bool // simplified bloom filter (exact for correctness)
}

// NewSegmentCache creates a new cache with the given capacity in bytes.
func NewSegmentCache(maxBytes int64) *SegmentCache {
	return &SegmentCache{
		entries:  make(map[string]*entry),
		lru:      list.New(),
		maxBytes: maxBytes,
		seenFilter: make(map[string]bool),
	}
}

// Get retrieves a value from the cache. Returns nil if not present.
func (c *SegmentCache) Get(key string) []byte {
	c.mu.Lock()
	defer c.mu.Unlock()

	if e, ok := c.entries[key]; ok {
		c.lru.MoveToFront(e.elem)
		return e.value
	}
	return nil
}

// Put stores a value in the cache with second-access admission control.
// On first access, the key is added to the seen filter but NOT cached.
// On second access, the key is admitted to the LRU cache.
// Returns true if the value was admitted to the cache.
func (c *SegmentCache) Put(key string, value []byte) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Already cached — update in place
	if e, ok := c.entries[key]; ok {
		c.currentBytes -= e.size
		e.value = value
		e.size = int64(len(value))
		c.currentBytes += e.size
		c.lru.MoveToFront(e.elem)
		return true
	}

	// Admission control: only cache on second access
	if !c.seenFilter[key] {
		c.seenFilter[key] = true
		return false
	}

	// Admit to cache
	size := int64(len(value))

	// Evict LRU entries until we have room
	for c.currentBytes+size > c.maxBytes && c.lru.Len() > 0 {
		c.evictLRU()
	}

	// If single entry exceeds max, don't cache
	if size > c.maxBytes {
		return false
	}

	e := &entry{key: key, value: value, size: size}
	e.elem = c.lru.PushFront(e)
	c.entries[key] = e
	c.currentBytes += size
	return true
}

// Remove explicitly removes a key from the cache and seen filter.
func (c *SegmentCache) Remove(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if e, ok := c.entries[key]; ok {
		c.lru.Remove(e.elem)
		c.currentBytes -= e.size
		delete(c.entries, key)
	}
	delete(c.seenFilter, key)
}

// Len returns the number of cached entries.
func (c *SegmentCache) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.entries)
}

// SizeBytes returns the current cache size in bytes.
func (c *SegmentCache) SizeBytes() int64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.currentBytes
}

// HitRate returns cache hit statistics.
func (c *SegmentCache) Stats() CacheStats {
	c.mu.Lock()
	defer c.mu.Unlock()
	return CacheStats{
		Entries:      len(c.entries),
		CurrentBytes: c.currentBytes,
		MaxBytes:     c.maxBytes,
		SeenKeys:     len(c.seenFilter),
	}
}

// CacheStats holds cache performance counters.
type CacheStats struct {
	Entries      int
	CurrentBytes int64
	MaxBytes     int64
	SeenKeys     int
}

// evictLRU removes the least recently used entry. Must hold c.mu.
func (c *SegmentCache) evictLRU() {
	back := c.lru.Back()
	if back == nil {
		return
	}
	e := back.Value.(*entry)
	c.lru.Remove(back)
	c.currentBytes -= e.size
	delete(c.entries, e.key)
}

// Clear removes all entries from the cache and resets the seen filter.
func (c *SegmentCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries = make(map[string]*entry)
	c.lru.Init()
	c.currentBytes = 0
	c.seenFilter = make(map[string]bool)
}
