package hnsw

import (
	"context"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/POTATO-VE1/Magnitude/internal/index/flat"
)

// TestHNSW_RecallHard is a stress test with 50K vectors at 768 dimensions.
// This simulates real CLIP/SigLIP embeddings where HNSW recall actually matters.
func TestHNSW_RecallHard(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping hard recall test in short mode")
	}
	const (
		dim            = 768
		numVectors     = 5000
		numQueries     = 50
		k              = 10
		metric         = "cosine"
	)

	// Use structured random data with clusters (more realistic than uniform)
	rng := rand.New(rand.NewSource(42))

	// Generate 100 cluster centroids
	numClusters := 100
	centroids := make([][]float32, numClusters)
	for i := range centroids {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rng.Float32()*2 - 1 // [-1, 1]
		}
		normalize(vec)
		centroids[i] = vec
	}

	// Generate vectors around centroids (clustered data is harder for HNSW)
	t.Logf("Generating %d vectors in %d clusters...", numVectors, numClusters)
	vectors := make([][]float32, numVectors)
	for i := range vectors {
		centroid := centroids[i%numClusters]
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = centroid[j] + rng.Float32()*0.3 - 0.15 // noise around centroid
		}
		normalize(vec)
		vectors[i] = vec
	}

	// Build ground truth with flat index (brute force)
	t.Log("Building flat index for ground truth...")
	flatStart := time.Now()
	flatIdx, _ := flat.NewFlatIndex(dim, metric)
	for i, vec := range vectors {
		flatIdx.Insert(uint64(i), vec)
	}
	t.Logf("Flat index built in %v", time.Since(flatStart))

	// Generate queries (from different clusters)
	queryRng := rand.New(rand.NewSource(99))
	queries := make([][]float32, numQueries)
	for q := range queries {
		centroid := centroids[q%numClusters]
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = centroid[j] + queryRng.Float32()*0.3 - 0.15
		}
		normalize(vec)
		queries[q] = vec
	}

	ctx := context.Background()

	// Compute ground truth
	t.Log("Computing ground truth...")
	gtStart := time.Now()
	gtResults := make([][]uint64, numQueries)
	for q, query := range queries {
		results, _ := flatIdx.Search(ctx, query, k, 0)
		ids := make([]uint64, k)
		for i, r := range results {
			ids[i] = r.ID
		}
		gtResults[q] = ids
	}
	t.Logf("Ground truth computed in %v", time.Since(gtStart))

	// Test configurations
	type config struct {
		m   int
		efC int
		efS int
	}

	configs := []config{
		// ChromaDB likely uses these or higher
		{16, 400, 50},
		{16, 400, 128},
		{16, 400, 200},
		{16, 400, 256},
		// Higher M
		{24, 400, 128},
		{24, 400, 200},
		{32, 400, 200},
		{32, 400, 256},
		{48, 400, 256},
		{48, 400, 400},
	}

	t.Logf("\n%-6s %-6s %-6s  %-10s  %-12s  %s", "M", "efC", "efS", "Recall@10", "Build Time", "Search QPS")
	t.Logf("%-6s %-6s %-6s  %-10s  %-12s  %s", "------", "------", "------", "----------", "------------", "----------")

	for _, cfg := range configs {
		// Build HNSW
		hnswIdx, err := NewHNSWIndex(dim, cfg.m, cfg.efC, cfg.efS, metric)
		if err != nil {
			t.Fatalf("NewHNSWIndex: %v", err)
		}

		buildStart := time.Now()
		for i, vec := range vectors {
			if err := hnswIdx.Insert(uint64(i), vec); err != nil {
				t.Fatalf("Insert(%d): %v", i, err)
			}
		}
		buildTime := time.Since(buildStart)

		// Drain dirty buffer before measuring recall
		hnswIdx.drainDirtyBuffer()

		// Measure recall
		var totalRecall float64
		searchStart := time.Now()
		for q, query := range queries {
			results, _ := hnswIdx.Search(ctx, query, k, 0)
			gtSet := make(map[uint64]bool, k)
			for _, id := range gtResults[q] {
				gtSet[id] = true
			}
			hits := 0
			for _, r := range results {
				if gtSet[r.ID] {
					hits++
				}
			}
			totalRecall += float64(hits) / float64(k)
		}
		searchTime := time.Since(searchStart)
		avgRecall := totalRecall / float64(numQueries)
		qps := float64(numQueries) / searchTime.Seconds()

		status := ""
		switch {
		case avgRecall >= 0.99:
			status = "*** ChromaDB-level"
		case avgRecall >= 0.985:
			status = "**  GOOD"
		case avgRecall >= 0.98:
			status = "*   OK"
		}

		t.Logf("%-6d %-6d %-6d  %-10.4f  %-12v  %.0f qps  %s",
			cfg.m, cfg.efC, cfg.efS, avgRecall, buildTime.Round(time.Second), qps, status)

		// Clean up
		hnswIdx.Close()
	}
}

func normalize(v []float32) {
	var norm float32
	for _, x := range v {
		norm += x * x
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 {
		for i := range v {
			v[i] /= norm
		}
	}
}
