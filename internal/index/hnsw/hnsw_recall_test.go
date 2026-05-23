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
		efConstruction = 400
		efSearch       = 128
		numVectors     = 1000
		numQueries     = 100
		k              = 10
		minRecall      = 0.98
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

func TestHNSW_Concurrency(t *testing.T) {
	const (
		dim            = 64
		m              = 16
		efConstruction = 100
		efSearch       = 50
		numVectors     = 200
		k              = 5
		metric         = "l2"
	)

	rng := rand.New(rand.NewSource(123))
	vectors := make([][]float32, numVectors)
	for i := range vectors {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		vectors[i] = vec
	}

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

	// Spin up concurrent readers
	const numWorkers = 20
	const queriesPerWorker = 50
	errChan := make(chan error, numWorkers)

	for w := 0; w < numWorkers; w++ {
		go func(workerID int) {
			localRng := rand.New(rand.NewSource(int64(456 + workerID)))
			for q := 0; q < queriesPerWorker; q++ {
				query := make([]float32, dim)
				for j := range query {
					query[j] = localRng.Float32()
				}
				_, err := hnswIdx.Search(ctx, query, k, 0)
				if err != nil {
					errChan <- err
					return
				}
			}
			errChan <- nil
		}(w)
	}

	for w := 0; w < numWorkers; w++ {
		if err := <-errChan; err != nil {
			t.Errorf("Concurrent search failed: %v", err)
		}
	}
}

func TestHNSW_SearchFiltered(t *testing.T) {
	const (
		dim            = 64
		m              = 16
		efConstruction = 200
		efSearch       = 50
		numVectors     = 500
		k              = 5
		metric         = "l2"
	)

	rng := rand.New(rand.NewSource(42))

	// Generate vectors with "category" metadata (0 or 1)
	vectors := make([][]float32, numVectors)
	categories := make([]int, numVectors)
	for i := range vectors {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		vectors[i] = vec
		categories[i] = i % 2 // 50% are category 0, 50% are category 1
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

	// Build flat index for ground truth (only category 0)
	flatIdx, err := flat.NewFlatIndex(dim, metric)
	if err != nil {
		t.Fatalf("NewFlatIndex: %v", err)
	}
	validIDs := make(map[uint64]bool)
	for i, vec := range vectors {
		if categories[i] == 0 {
			flatIdx.Insert(uint64(i), vec)
			validIDs[uint64(i)] = true
		}
	}

	// Test filtered search
	query := vectors[0] // use first vector as query
	ctx := context.Background()

	filteredResults, err := hnswIdx.SearchFiltered(ctx, query, k, efSearch, validIDs)
	if err != nil {
		t.Fatalf("SearchFiltered: %v", err)
	}

	// Get ground truth from flat index
	groundTruth, err := flatIdx.Search(ctx, query, k, 0)
	if err != nil {
		t.Fatalf("flat.Search: %v", err)
	}

	// Verify all filtered results are in the valid set
	for _, r := range filteredResults {
		if !validIDs[r.ID] {
			t.Errorf("SearchFiltered returned ID %d which is not in valid set", r.ID)
		}
	}

	// Verify we got results
	if len(filteredResults) == 0 {
		t.Error("SearchFiltered returned 0 results")
	}

	// Check recall (should be high since filter is 50%)
	matches := 0
	for _, r := range filteredResults {
		for _, gt := range groundTruth {
			if r.ID == gt.ID {
				matches++
				break
			}
		}
	}
	recall := float64(matches) / float64(len(groundTruth))
	t.Logf("Filtered search recall: %.2f (%d/%d matches)", recall, matches, len(groundTruth))
	if recall < 0.80 {
		t.Errorf("recall %.2f < 0.80", recall)
	}
}

func TestHNSW_SearchFiltered_EmptyFilter(t *testing.T) {
	hnswIdx, err := NewHNSWIndex(64, 16, 200, 50, "l2")
	if err != nil {
		t.Fatalf("NewHNSWIndex: %v", err)
	}
	hnswIdx.Insert(1, make([]float32, 64))

	results, err := hnswIdx.SearchFiltered(context.Background(), make([]float32, 64), 5, 50, map[uint64]bool{})
	if err != nil {
		t.Fatalf("SearchFiltered: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results for empty filter, got %d", len(results))
	}
}
