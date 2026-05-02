package observability

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHTTPMetrics_Registered(t *testing.T) {
	// Verify that all HTTP metrics are registered with the default registry.
	// promauto registers automatically, so we just verify they're non-nil.
	assert.NotNil(t, HTTPRequestsTotal)
	assert.NotNil(t, HTTPRequestDuration)
	assert.NotNil(t, HTTPRequestSize)
	assert.NotNil(t, HTTPActiveRequests)
}

func TestIndexMetrics_Registered(t *testing.T) {
	assert.NotNil(t, IndexOperationsTotal)
	assert.NotNil(t, IndexOperationDuration)
	assert.NotNil(t, SearchRecallGauge)
	assert.NotNil(t, SearchResultsReturned)
}

func TestCollectionMetrics_Registered(t *testing.T) {
	assert.NotNil(t, CollectionsActive)
	assert.NotNil(t, VectorsTotal)
}

func TestStorageMetrics_Registered(t *testing.T) {
	assert.NotNil(t, WALOperationsTotal)
	assert.NotNil(t, WALSizeBytes)
	assert.NotNil(t, CompactionsTotal)
}

func TestHTTPMetrics_Increment(t *testing.T) {
	// Test that counter increment doesn't panic
	HTTPRequestsTotal.WithLabelValues("GET", "/v1/health", "200").Inc()
	HTTPActiveRequests.Inc()
	HTTPActiveRequests.Dec()
}

func TestIndexMetrics_Observe(t *testing.T) {
	IndexOperationsTotal.WithLabelValues("flat", "search").Inc()
	IndexOperationDuration.WithLabelValues("flat", "search").Observe(0.005)
	SearchResultsReturned.WithLabelValues("flat").Observe(10)
	SearchRecallGauge.WithLabelValues("flat", "test-collection").Set(0.95)
}

func TestCollectionMetrics_Set(t *testing.T) {
	CollectionsActive.Set(5)
	VectorsTotal.WithLabelValues("col-1", "my-collection").Set(10000)
}

func TestStorageMetrics_Increment(t *testing.T) {
	WALOperationsTotal.WithLabelValues("insert").Inc()
	WALSizeBytes.Set(1024 * 1024)
	CompactionsTotal.Inc()
}

func TestMetrics_GatherAll(t *testing.T) {
	// Ensure all metrics have at least one data point
	HTTPRequestsTotal.WithLabelValues("GET", "/test", "200").Inc()
	HTTPRequestDuration.WithLabelValues("GET", "/test").Observe(0.001)
	HTTPRequestSize.WithLabelValues("GET", "/test").Observe(100)
	HTTPActiveRequests.Set(0)
	IndexOperationsTotal.WithLabelValues("flat", "search").Inc()
	IndexOperationDuration.WithLabelValues("flat", "search").Observe(0.005)
	CollectionsActive.Set(1)
	WALOperationsTotal.WithLabelValues("insert").Inc()

	// Verify all metrics can be gathered without error
	families, err := prometheus.DefaultGatherer.Gather()
	require.NoError(t, err)
	assert.Greater(t, len(families), 0, "should have registered metrics families")

	// Check that our metrics are present
	metricNames := make(map[string]bool)
	for _, f := range families {
		metricNames[f.GetName()] = true
	}

	expectedMetrics := []string{
		"vectordb_http_requests_total",
		"vectordb_http_request_duration_seconds",
		"vectordb_index_operations_total",
		"vectordb_collection_active",
		"vectordb_wal_operations_total",
	}

	for _, name := range expectedMetrics {
		assert.True(t, metricNames[name], "metric %q should be registered", name)
	}
}
