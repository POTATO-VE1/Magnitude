package hnsw

import (
	"context"
	"math/rand"
	"testing"

	"github.com/POTATO-VE1/Magnitude/internal/index/flat"
)

func TestHNSW_RecallAt10(t *testing.T) {
	const (
		dim            = 128
		m              = 16
		efConstruction = 200
		efSearch       = 50
		numVectors     = 1000
		numQueries     = 100
		k              = 10
		minRecall      = 0.90
		metric         = "l2"
	)

	rng := rand.New(rand.NewSource(42))

	// Generate random vectors
	vectors := make([][]float32, numVectors)
	for i := range vectors {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		vectors[i] = vec
	}

	// Build flat index (ground truth)
	flatIdx, err := flat.NewFlatIndex(dim, metric)
	if err != nil {
		t.Fatalf("NewFlatIndex: %v", err)
	}
	for i, vec := range vectors {
		if err := flatIdx.Insert(uint64(i), vec); err != nil {
			t.Fatalf("flat.Insert(%d): %v", i, err)
		}
	}

	// Build HNSW index
	hnswIdx, err := NewHNSWIndex(dim, m, efConstruction, efSearch, metric)
	if err != nil {
		t.Fatalf("NewHNSWIndex: %v", err)
	}
	for i, vec := range vectors {
		if err := hnswIdx.Insert(uint64(i), vec); err != nil {
			t.Fatalf("hnsw.Insert(%d): %v", i, err)
		}
	}

	ctx := context.Background()

	// Generate random query vectors and measure recall
	queryRng := rand.New(rand.NewSource(99))
	var totalRecall float64

	for q := 0; q < numQueries; q++ {
		query := make([]float32, dim)
		for j := range query {
			query[j] = queryRng.Float32()
		}

		// Ground truth from flat index
		gtResults, err := flatIdx.Search(ctx, query, k, 0)
		if err != nil {
			t.Fatalf("flat.Search(query %d): %v", q, err)
		}

		// HNSW approximate search
		hnswResults, err := hnswIdx.Search(ctx, query, k, 0)
		if err != nil {
			t.Fatalf("hnsw.Search(query %d): %v", q, err)
		}

		// Build set of ground truth IDs
		gtIDs := make(map[uint64]struct{}, len(gtResults))
		for _, r := range gtResults {
			gtIDs[r.ID] = struct{}{}
		}

		// Count intersection
		var hits int
		for _, r := range hnswResults {
			if _, ok := gtIDs[r.ID]; ok {
				hits++
			}
		}

		recall := float64(hits) / float64(k)
		totalRecall += recall
	}

	avgRecall := totalRecall / float64(numQueries)
	t.Logf("Average recall@%d over %d queries: %.4f (threshold: %.2f)", k, numQueries, avgRecall, minRecall)

	if avgRecall < minRecall {
		t.Errorf("recall %.4f < threshold %.2f", avgRecall, minRecall)
	}
}
