// Package observability provides Prometheus metrics for the VectorDB server.
//
// Metric naming follows the Prometheus best practices:
//   - vectordb_ prefix for all metrics
//   - _total suffix for counters
//   - _seconds suffix for duration histograms
//   - _bytes suffix for size measurements
//
// Three categories of metrics:
//   1. HTTP API metrics — request counts, latencies, status codes
//   2. Index metrics — insert/search/delete counts, latencies
//   3. System metrics — active collections, total vectors, goroutines
package observability

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// ── HTTP API Metrics ────────────────────────────────────────────────────────

var (
	// HTTPRequestsTotal counts HTTP requests by method, path, and status code.
	HTTPRequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "vectordb",
			Subsystem: "http",
			Name:      "requests_total",
			Help:      "Total number of HTTP requests by method, path, and status code.",
		},
		[]string{"method", "path", "status"},
	)

	// HTTPRequestDuration observes request latency by method and path.
	HTTPRequestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "vectordb",
			Subsystem: "http",
			Name:      "request_duration_seconds",
			Help:      "HTTP request duration in seconds.",
			Buckets:   []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0},
		},
		[]string{"method", "path"},
	)

	// HTTPRequestSize observes request body size.
	HTTPRequestSize = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "vectordb",
			Subsystem: "http",
			Name:      "request_size_bytes",
			Help:      "HTTP request body size in bytes.",
			Buckets:   prometheus.ExponentialBuckets(100, 10, 6), // 100B, 1KB, 10KB, 100KB, 1MB, 10MB
		},
		[]string{"method", "path"},
	)

	// HTTPActiveRequests tracks currently in-flight requests.
	HTTPActiveRequests = promauto.NewGauge(
		prometheus.GaugeOpts{
			Namespace: "vectordb",
			Subsystem: "http",
			Name:      "active_requests",
			Help:      "Number of currently active HTTP requests.",
		},
	)
)

// ── Index Metrics ───────────────────────────────────────────────────────────

var (
	// IndexOperationsTotal counts index operations by collection, type, and operation.
	IndexOperationsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "vectordb",
			Subsystem: "index",
			Name:      "operations_total",
			Help:      "Total number of index operations by type and operation.",
		},
		[]string{"index_type", "operation"}, // operation: insert, search, delete, rebuild
	)

	// IndexOperationDuration observes index operation latency.
	IndexOperationDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "vectordb",
			Subsystem: "index",
			Name:      "operation_duration_seconds",
			Help:      "Index operation duration in seconds.",
			Buckets:   []float64{0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0},
		},
		[]string{"index_type", "operation"},
	)

	// SearchRecallGauge tracks the last measured recall@K for monitoring index quality.
	SearchRecallGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "vectordb",
			Subsystem: "index",
			Name:      "search_recall",
			Help:      "Last measured recall@K for the index.",
		},
		[]string{"index_type", "collection_id"},
	)

	// SearchResultsReturned observes the number of results returned per search.
	SearchResultsReturned = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "vectordb",
			Subsystem: "index",
			Name:      "search_results_returned",
			Help:      "Number of results returned per search.",
			Buckets:   []float64{1, 5, 10, 20, 50, 100},
		},
		[]string{"index_type"},
	)
)

// ── Collection Metrics ──────────────────────────────────────────────────────

var (
	// CollectionsActive tracks the number of active collections.
	CollectionsActive = promauto.NewGauge(
		prometheus.GaugeOpts{
			Namespace: "vectordb",
			Subsystem: "collection",
			Name:      "active",
			Help:      "Number of active collections.",
		},
	)

	// VectorsTotal tracks the total number of vectors across all collections.
	VectorsTotal = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "vectordb",
			Subsystem: "collection",
			Name:      "vectors_total",
			Help:      "Total number of vectors in a collection.",
		},
		[]string{"collection_id", "collection_name"},
	)
)

// ── Storage Metrics ─────────────────────────────────────────────────────────

var (
	// WALOperationsTotal counts WAL operations.
	WALOperationsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "vectordb",
			Subsystem: "wal",
			Name:      "operations_total",
			Help:      "Total WAL operations by type.",
		},
		[]string{"operation"}, // insert, delete
	)

	// WALSizeBytes tracks current WAL size.
	WALSizeBytes = promauto.NewGauge(
		prometheus.GaugeOpts{
			Namespace: "vectordb",
			Subsystem: "wal",
			Name:      "size_bytes",
			Help:      "Current WAL database size in bytes.",
		},
	)

	// CompactionsTotal counts compaction runs.
	CompactionsTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Namespace: "vectordb",
			Subsystem: "storage",
			Name:      "compactions_total",
			Help:      "Total number of compaction runs.",
		},
	)
)
