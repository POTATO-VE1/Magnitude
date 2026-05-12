// Package security — Per-IP token bucket rate limiter.
//
// Rate limiting model:
//  1. Per-IP: prevents a single IP from overwhelming the server.
//
// Implementation uses golang.org/x/time/rate (token bucket algorithm):
//   - tokens are added at rate r tokens/second
//   - burst allows up to b tokens to be consumed instantaneously
//   - when tokens are exhausted, requests receive HTTP 429
//
// Cleanup: A background goroutine evicts rate limiter entries for IPs
// that have been idle for > 5 minutes, preventing unbounded memory growth.
package security

import (
	"context"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/POTATO-VE1/Magnitude/internal/metadata"
	"golang.org/x/time/rate"
)

// RateLimiter provides per-IP rate limiting using token buckets.
type RateLimiter struct {
	mu       sync.RWMutex
	limiters map[string]*limiterEntry
	rps      rate.Limit
	burst    int
	cancel   context.CancelFunc
	done     chan struct{}
}

type limiterEntry struct {
	limiter  *rate.Limiter
	lastSeen time.Time
}

// NewRateLimiter creates a rate limiter with the given requests/second and burst.
// Starts a background goroutine that evicts idle entries every 3 minutes.
func NewRateLimiter(rps float64, burst int) *RateLimiter {
	ctx, cancel := context.WithCancel(context.Background())
	rl := &RateLimiter{
		limiters: make(map[string]*limiterEntry),
		rps:      rate.Limit(rps),
		burst:    burst,
		cancel:   cancel,
		done:     make(chan struct{}),
	}
	go rl.cleanupLoop(ctx)
	return rl
}

// Allow checks if a request from the given IP is allowed.
func (rl *RateLimiter) Allow(r *http.Request) bool {
	ip := extractIP(r)
	limiter := rl.getLimiter(ip)
	return limiter.Allow()
}

// getLimiter returns the rate limiter for the given IP, creating one if needed.
func (rl *RateLimiter) getLimiter(ip string) *rate.Limiter {
	// Fast path: read lock
	rl.mu.RLock()
	entry, exists := rl.limiters[ip]
	rl.mu.RUnlock()

	if exists {
		rl.mu.Lock()
		entry.lastSeen = time.Now()
		rl.mu.Unlock()
		return entry.limiter
	}

	// Slow path: write lock
	rl.mu.Lock()
	defer rl.mu.Unlock()

	// Double-check under write lock
	if entry, exists := rl.limiters[ip]; exists {
		entry.lastSeen = time.Now()
		return entry.limiter
	}

	limiter := rate.NewLimiter(rl.rps, rl.burst)
	rl.limiters[ip] = &limiterEntry{
		limiter:  limiter,
		lastSeen: time.Now(),
	}
	return limiter
}

// cleanupLoop evicts idle entries every 3 minutes.
func (rl *RateLimiter) cleanupLoop(ctx context.Context) {
	defer close(rl.done)
	ticker := time.NewTicker(3 * time.Minute)
	defer ticker.Stop()

	const idleTimeout = 5 * time.Minute

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			rl.mu.Lock()
			now := time.Now()
			for ip, entry := range rl.limiters {
				if now.Sub(entry.lastSeen) > idleTimeout {
					delete(rl.limiters, ip)
				}
			}
			rl.mu.Unlock()
		}
	}
}

// Stop halts the cleanup goroutine.
func (rl *RateLimiter) Stop() {
	rl.cancel()
	<-rl.done
}

// extractIP extracts the client IP, only trusting X-Forwarded-For from
// trusted proxies (localhost/private network). Direct clients cannot spoof
// their IP via X-Forwarded-For because RemoteAddr is checked first.
func extractIP(r *http.Request) string {
	if isTrustedProxy(r.RemoteAddr) {
		if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
			if idx := strings.IndexByte(xff, ','); idx >= 0 {
				return strings.TrimSpace(xff[:idx])
			}
			return strings.TrimSpace(xff)
		}
	}

	ip, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return ip
}

// isTrustedProxy returns true if the remote address is from a trusted proxy
// (loopback or private network). Only trusted proxies' X-Forwarded-For headers
// are honored to prevent IP spoofing.
func isTrustedProxy(remoteAddr string) bool {
	host, _, err := net.SplitHostPort(remoteAddr)
	if err != nil {
		host = remoteAddr
	}
	ip := net.ParseIP(host)
	if ip == nil {
		return false
	}
	return ip.IsLoopback() || ip.IsPrivate()
}

// ── Multi-Tenancy Rate Limiter ──────────────────────────────────────────────

// TenantRateLimiter provides per-tenant rate limiting using quotas from SysDB.
type TenantRateLimiter struct {
	mu       sync.Mutex
	limiters map[string]*rate.Limiter // tenantID → limiter
	sysdb    *metadata.SysDB
}

// NewTenantRateLimiter creates a new TenantRateLimiter.
func NewTenantRateLimiter(sysdb *metadata.SysDB) *TenantRateLimiter {
	return &TenantRateLimiter{
		limiters: make(map[string]*rate.Limiter),
		sysdb:    sysdb,
	}
}

// getOrCreate returns the rate limiter for a tenant, fetching the quota if needed.
func (rl *TenantRateLimiter) getOrCreate(tenantID string) *rate.Limiter {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	if limiter, exists := rl.limiters[tenantID]; exists {
		return limiter
	}

	// Default fallback quota
	rps := 100.0
	burst := 100

	// Fetch quota from SysDB
	quotas, err := rl.sysdb.GetTenantQuotas(tenantID)
	if err == nil && quotas != nil && quotas.MaxQPS > 0 {
		rps = float64(quotas.MaxQPS)
		burst = quotas.MaxQPS
	}

	limiter := rate.NewLimiter(rate.Limit(rps), burst)
	rl.limiters[tenantID] = limiter
	return limiter
}

// Middleware enforces the rate limit for the authenticated tenant.
func (rl *TenantRateLimiter) Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tenantID := TenantID(r.Context())
		if tenantID == "" {
			// Pass through if not authenticated (or health checks)
			next.ServeHTTP(w, r)
			return
		}

		limiter := rl.getOrCreate(tenantID)
		if !limiter.Allow() {
			w.Header().Set("Retry-After", "1")
			http.Error(w, `{"error":"tenant rate limit exceeded"}`, http.StatusTooManyRequests)
			return
		}
		next.ServeHTTP(w, r)
	})
}
