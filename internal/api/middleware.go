// Package api — HTTP middleware stack.
//
// Middleware chain (applied in order, outermost first):
//  1. Recovery       — catch panics, return 500, log stack trace
//  2. RequestID      — generate/propagate X-Request-ID for tracing
//  3. StructuredLog  — log method, path, status, duration via slog
//  4. Metrics        — increment Prometheus counters, observe latency histograms
//  5. Auth           — validate Bearer token via SHA-256 hash lookup
//  6. RateLimit      — per-IP token bucket; returns 429 on exhaustion
//  7. MaxBodySize    — limit request body to prevent OOM on malformed input
package api

import (
	"fmt"
	"log/slog"
	"net/http"
	"runtime/debug"
	"time"

	"github.com/google/uuid"
	obs "github.com/POTATO-VE1/Magnitude/internal/observability"
)

// responseWriter wraps http.ResponseWriter to capture the status code.
type responseWriter struct {
	http.ResponseWriter
	statusCode int
	written    bool
}

func (rw *responseWriter) WriteHeader(code int) {
	if !rw.written {
		rw.statusCode = code
		rw.written = true
	}
	rw.ResponseWriter.WriteHeader(code)
}

func (rw *responseWriter) Write(b []byte) (int, error) {
	if !rw.written {
		rw.statusCode = http.StatusOK
		rw.written = true
	}
	return rw.ResponseWriter.Write(b)
}

// Recovery middleware catches panics, logs the stack trace, and returns 500.
func Recovery(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				slog.Error("panic recovered",
					"error", fmt.Sprintf("%v", err),
					"stack", string(debug.Stack()),
					"method", r.Method,
					"path", r.URL.Path,
				)
				http.Error(w, `{"error":"internal server error"}`, http.StatusInternalServerError)
			}
		}()
		next.ServeHTTP(w, r)
	})
}

// RequestID middleware generates or propagates an X-Request-ID header.
func RequestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reqID := r.Header.Get("X-Request-ID")
		if reqID == "" {
			reqID = uuid.New().String()
		}
		w.Header().Set("X-Request-ID", reqID)
		r.Header.Set("X-Request-ID", reqID)
		next.ServeHTTP(w, r)
	})
}

// StructuredLog middleware logs request method, path, status code, and duration.
func StructuredLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rw := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		next.ServeHTTP(rw, r)

		duration := time.Since(start)
		slog.Info("request",
			"method", r.Method,
			"path", r.URL.Path,
			"status", rw.statusCode,
			"duration_ms", duration.Milliseconds(),
			"request_id", r.Header.Get("X-Request-ID"),
			"remote_addr", r.RemoteAddr,
		)
	})
}



// MaxBodySize middleware limits the request body to maxBytes.
func MaxBodySize(maxBytes int64) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			r.Body = http.MaxBytesReader(w, r.Body, maxBytes)
			next.ServeHTTP(w, r)
		})
	}
}

// Metrics middleware instruments HTTP requests with Prometheus counters and histograms.
func Metrics(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		obs.HTTPActiveRequests.Inc()
		defer obs.HTTPActiveRequests.Dec()

		start := time.Now()
		rw := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		next.ServeHTTP(rw, r)

		duration := time.Since(start).Seconds()
		status := fmt.Sprintf("%d", rw.statusCode)
		path := r.URL.Path

		obs.HTTPRequestsTotal.WithLabelValues(r.Method, path, status).Inc()
		obs.HTTPRequestDuration.WithLabelValues(r.Method, path).Observe(duration)

		if r.ContentLength > 0 {
			obs.HTTPRequestSize.WithLabelValues(r.Method, path).Observe(float64(r.ContentLength))
		}
	})
}

