// Package storage — Admission-controlled object storage wrapper.
//
// ChromaDB's distributed storage uses an admission control layer
// (admission_controlled_s3) that prevents S3 503 SlowDown errors
// by tracking in-flight requests per S3 prefix and queuing excess
// requests.
//
// S3 enforces soft rate limits per key prefix (~5,500 GET/s, ~3,500 PUT/s
// per prefix). Without admission control, a compaction storm or cache miss
// burst can trigger rate limiting and cascade into latency spikes.
//
// This implementation is generic — it works with any object storage backend
// (S3, GCS, Azure Blob, MinIO) via the ObjectStore interface.
package storage

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ObjectStore is the interface for object storage backends.
// Implement this for S3, GCS, Azure Blob, or local filesystem.
type ObjectStore interface {
	Get(ctx context.Context, bucket, key string) ([]byte, error)
	Put(ctx context.Context, bucket, key string, data []byte) error
	Delete(ctx context.Context, bucket, key string) error
	// ConditionalPut writes data only if the current ETag matches.
	// Returns the new ETag on success, ErrPreconditionFailed if ETag mismatch.
	ConditionalPut(ctx context.Context, bucket, key string, data []byte, expectedETag string) (string, error)
}

// AdmissionController limits concurrent access to object storage per prefix.
// Prevents S3 503 SlowDown errors under high load.
type AdmissionController struct {
	mu           sync.Mutex
	inFlight     map[string]int           // prefix → count of in-flight requests
	maxPerPrefix int                      // max concurrent requests per prefix
	waitQueues   map[string][]chan struct{} // prefix → waiters

	// Observability
	totalAcquires atomic.Int64
	totalWaits    atomic.Int64
	totalTimeouts atomic.Int64
}

// NewAdmissionController creates a new admission controller.
// maxPerPrefix limits concurrent requests per S3 key prefix.
// Recommended: 300 for S3 (well below 3,500 PUT/s limit with headroom).
func NewAdmissionController(maxPerPrefix int) *AdmissionController {
	return &AdmissionController{
		inFlight:     make(map[string]int),
		maxPerPrefix: maxPerPrefix,
		waitQueues:   make(map[string][]chan struct{}),
	}
}

// Acquire blocks until a slot is available for the given prefix.
// Returns an error if ctx is cancelled while waiting.
func (ac *AdmissionController) Acquire(ctx context.Context, prefix string) error {
	ac.totalAcquires.Add(1)

	ac.mu.Lock()
	if ac.inFlight[prefix] < ac.maxPerPrefix {
		ac.inFlight[prefix]++
		ac.mu.Unlock()
		return nil
	}

	// Queue this request
	ac.totalWaits.Add(1)
	ch := make(chan struct{}, 1)
	ac.waitQueues[prefix] = append(ac.waitQueues[prefix], ch)
	ac.mu.Unlock()

	// Wait for slot or context cancellation
	select {
	case <-ch:
		return nil
	case <-ctx.Done():
		ac.totalTimeouts.Add(1)
		// Remove ourselves from the wait queue
		ac.mu.Lock()
		q := ac.waitQueues[prefix]
		for i, c := range q {
			if c == ch {
				ac.waitQueues[prefix] = append(q[:i], q[i+1:]...)
				break
			}
		}
		ac.mu.Unlock()
		return ctx.Err()
	}
}

// Release releases a slot for the given prefix.
// Must be called after each Acquire (typically via defer).
func (ac *AdmissionController) Release(prefix string) {
	ac.mu.Lock()
	defer ac.mu.Unlock()

	if q := ac.waitQueues[prefix]; len(q) > 0 {
		// Wake up next waiter — they inherit the slot
		ch := q[0]
		ac.waitQueues[prefix] = q[1:]
		ch <- struct{}{}
	} else {
		ac.inFlight[prefix]--
		if ac.inFlight[prefix] == 0 {
			delete(ac.inFlight, prefix)
		}
	}
}

// InFlight returns the current in-flight count for a prefix.
func (ac *AdmissionController) InFlight(prefix string) int {
	ac.mu.Lock()
	defer ac.mu.Unlock()
	return ac.inFlight[prefix]
}

// QueueDepth returns the number of waiters for a prefix.
func (ac *AdmissionController) QueueDepth(prefix string) int {
	ac.mu.Lock()
	defer ac.mu.Unlock()
	return len(ac.waitQueues[prefix])
}

// Stats returns admission controller statistics.
func (ac *AdmissionController) Stats() AdmissionStats {
	ac.mu.Lock()
	totalInFlight := 0
	totalQueued := 0
	for _, v := range ac.inFlight {
		totalInFlight += v
	}
	for _, q := range ac.waitQueues {
		totalQueued += len(q)
	}
	ac.mu.Unlock()

	return AdmissionStats{
		TotalAcquires: ac.totalAcquires.Load(),
		TotalWaits:    ac.totalWaits.Load(),
		TotalTimeouts: ac.totalTimeouts.Load(),
		InFlight:      totalInFlight,
		Queued:        totalQueued,
	}
}

// AdmissionStats holds admission controller performance counters.
type AdmissionStats struct {
	TotalAcquires int64
	TotalWaits    int64
	TotalTimeouts int64
	InFlight      int
	Queued        int
}

// ── Admitted Object Store ──────────────────────────────────────────────────

// AdmittedStore wraps an ObjectStore with per-prefix admission control.
type AdmittedStore struct {
	inner      ObjectStore
	controller *AdmissionController
}

// NewAdmittedStore wraps an object store with admission control.
func NewAdmittedStore(inner ObjectStore, maxPerPrefix int) *AdmittedStore {
	return &AdmittedStore{
		inner:      inner,
		controller: NewAdmissionController(maxPerPrefix),
	}
}

// Get fetches an object with admission control.
func (s *AdmittedStore) Get(ctx context.Context, bucket, key string) ([]byte, error) {
	prefix := KeyPrefix(key)
	if err := s.controller.Acquire(ctx, prefix); err != nil {
		return nil, fmt.Errorf("admission: %w", err)
	}
	defer s.controller.Release(prefix)
	return s.inner.Get(ctx, bucket, key)
}

// Put stores an object with admission control.
func (s *AdmittedStore) Put(ctx context.Context, bucket, key string, data []byte) error {
	prefix := KeyPrefix(key)
	if err := s.controller.Acquire(ctx, prefix); err != nil {
		return fmt.Errorf("admission: %w", err)
	}
	defer s.controller.Release(prefix)
	return s.inner.Put(ctx, bucket, key, data)
}

// Delete removes an object with admission control.
func (s *AdmittedStore) Delete(ctx context.Context, bucket, key string) error {
	prefix := KeyPrefix(key)
	if err := s.controller.Acquire(ctx, prefix); err != nil {
		return fmt.Errorf("admission: %w", err)
	}
	defer s.controller.Release(prefix)
	return s.inner.Delete(ctx, bucket, key)
}

// ConditionalPut stores an object with ETag-based CAS and admission control.
func (s *AdmittedStore) ConditionalPut(ctx context.Context, bucket, key string, data []byte, expectedETag string) (string, error) {
	prefix := KeyPrefix(key)
	if err := s.controller.Acquire(ctx, prefix); err != nil {
		return "", fmt.Errorf("admission: %w", err)
	}
	defer s.controller.Release(prefix)
	return s.inner.ConditionalPut(ctx, bucket, key, data, expectedETag)
}

// AdmissionStats returns the underlying admission controller stats.
func (s *AdmittedStore) AdmissionStats() AdmissionStats {
	return s.controller.Stats()
}

// KeyPrefix extracts the S3 prefix from a key.
// Uses the first two path components for rate-limit bucketing.
// Example: "logs/tenant1/db1/collection1/frag.parquet" → "logs/tenant1"
func KeyPrefix(key string) string {
	parts := strings.SplitN(key, "/", 4)
	if len(parts) >= 3 {
		return parts[0] + "/" + parts[1]
	}
	if len(parts) >= 2 {
		return parts[0] + "/" + parts[1]
	}
	return key
}

// ── In-Memory Object Store (for testing) ───────────────────────────────────

// MemoryStore is an in-memory object store for testing.
type MemoryStore struct {
	mu      sync.RWMutex
	objects map[string][]byte // "bucket/key" → data
	etags   map[string]string // "bucket/key" → etag
	latency time.Duration     // simulated latency per operation
}

// NewMemoryStore creates a new in-memory object store.
func NewMemoryStore(latency time.Duration) *MemoryStore {
	return &MemoryStore{
		objects: make(map[string][]byte),
		etags:   make(map[string]string),
		latency: latency,
	}
}

func (m *MemoryStore) fullKey(bucket, key string) string {
	return bucket + "/" + key
}

func (m *MemoryStore) Get(ctx context.Context, bucket, key string) ([]byte, error) {
	if m.latency > 0 {
		time.Sleep(m.latency)
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	data, ok := m.objects[m.fullKey(bucket, key)]
	if !ok {
		return nil, fmt.Errorf("object not found: %s/%s", bucket, key)
	}
	cp := make([]byte, len(data))
	copy(cp, data)
	return cp, nil
}

func (m *MemoryStore) Put(ctx context.Context, bucket, key string, data []byte) error {
	if m.latency > 0 {
		time.Sleep(m.latency)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	fk := m.fullKey(bucket, key)
	cp := make([]byte, len(data))
	copy(cp, data)
	m.objects[fk] = cp
	m.etags[fk] = fmt.Sprintf("etag-%d", time.Now().UnixNano())
	return nil
}

func (m *MemoryStore) Delete(ctx context.Context, bucket, key string) error {
	if m.latency > 0 {
		time.Sleep(m.latency)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	fk := m.fullKey(bucket, key)
	delete(m.objects, fk)
	delete(m.etags, fk)
	return nil
}

func (m *MemoryStore) ConditionalPut(ctx context.Context, bucket, key string, data []byte, expectedETag string) (string, error) {
	if m.latency > 0 {
		time.Sleep(m.latency)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	fk := m.fullKey(bucket, key)
	currentETag := m.etags[fk]
	if currentETag != "" && currentETag != expectedETag {
		return "", fmt.Errorf("precondition failed: expected etag %q, got %q", expectedETag, currentETag)
	}
	cp := make([]byte, len(data))
	copy(cp, data)
	m.objects[fk] = cp
	newETag := fmt.Sprintf("etag-%d", time.Now().UnixNano())
	m.etags[fk] = newETag
	return newETag, nil
}
