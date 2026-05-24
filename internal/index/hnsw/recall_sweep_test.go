package hnsw

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/POTATO-VE1/Magnitude/internal/index/flat"
)

func TestHNSW_RecallSweep(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping recall sweep in short mode")
	}
	const (
		dim        = 128
		numVectors = 1000
		numQueries = 100
		k          = 10
		metric     = "l2"
	)

	rng := rand.New(rand.NewSource(42))
	vectors := make([][]float32, numVectors)
	for i := range vectors {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		vectors[i] = vec
	}

	// Build ground truth
	flatIdx, _ := flat.NewFlatIndex(dim, metric)
	for i, vec := range vectors {
		flatIdx.Insert(uint64(i), vec)
	}

	queryRng := rand.New(rand.NewSource(99))
	queries := make([][]float32, numQueries)
	for q := range queries {
		query := make([]float32, dim)
		for j := range query {
			query[j] = queryRng.Float32()
		}
		queries[q] = query
	}

	ctx := context.Background()
	gtResults := make([][]uint64, numQueries)
	for q, query := range queries {
		results, _ := flatIdx.Search(ctx, query, k, 0)
		ids := make([]uint64, k)
		for i, r := range results {
			ids[i] = r.ID
		}
		gtResults[q] = ids
	}

	// Sweep parameters
	type params struct {
		m   int
		efC int
		efS int
	}

	configs := []params{
		{16, 200, 50},
		{16, 200, 100},
		{16, 200, 128},
		{16, 200, 200},
		{16, 200, 256},
		{16, 400, 50},
		{16, 400, 100},
		{16, 400, 128},
		{16, 400, 200},
		{16, 400, 256},
		{16, 400, 400},
		{24, 200, 128},
		{24, 200, 200},
		{24, 400, 128},
		{24, 400, 200},
		{32, 400, 200},
		{32, 400, 256},
		{48, 400, 256},
	}

	t.Logf("%-6s %-6s %-6s  %s  %s", "M", "efC", "efS", "Recall", "Status")
	t.Logf("%-6s %-6s %-6s  %s  %s", "------", "------", "------", "------", "------")

	for _, p := range configs {
		hnswIdx, err := NewHNSWIndex(dim, p.m, p.efC, p.efS, metric)
		if err != nil {
			t.Fatalf("NewHNSWIndex(m=%d,efC=%d,efS=%d): %v", p.m, p.efC, p.efS, err)
		}
		for i, vec := range vectors {
			hnswIdx.Insert(uint64(i), vec)
		}

		var totalRecall float64
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
		avgRecall := totalRecall / float64(numQueries)

		status := ""
		switch {
		case avgRecall >= 0.99:
			status = "*** BEST"
		case avgRecall >= 0.985:
			status = "**  GOOD"
		case avgRecall >= 0.98:
			status = "*   OK"
		default:
			status = ""
		}

		t.Logf("%-6d %-6d %-6d  %.4f  %s", p.m, p.efC, p.efS, avgRecall, status)
	}
}

// TestHNSW_RecallVsLatency measures recall and approximate search latency
// for each parameter combination to help pick the best tradeoff.
func TestHNSW_RecallVsLatency(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping recall vs latency in short mode")
	}
	const (
		dim        = 128
		numVectors = 5000
		numQueries = 200
		k          = 10
		metric     = "l2"
	)

	rng := rand.New(rand.NewSource(42))
	vectors := make([][]float32, numVectors)
	for i := range vectors {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		vectors[i] = vec
	}

	flatIdx, _ := flat.NewFlatIndex(dim, metric)
	for i, vec := range vectors {
		flatIdx.Insert(uint64(i), vec)
	}

	queryRng := rand.New(rand.NewSource(99))
	queries := make([][]float32, numQueries)
	for q := range queries {
		query := make([]float32, dim)
		for j := range query {
			query[j] = queryRng.Float32()
		}
		queries[q] = query
	}

	ctx := context.Background()
	gtResults := make([][]uint64, numQueries)
	for q, query := range queries {
		results, _ := flatIdx.Search(ctx, query, k, 0)
		ids := make([]uint64, k)
		for i, r := range results {
			ids[i] = r.ID
		}
		gtResults[q] = ids
	}

	type result struct {
		m      int
		efS    int
		recall float64
		qps    float64
	}

	configs := []struct{ m, efC, efS int }{
		{16, 400, 50},
		{16, 400, 100},
		{16, 400, 128},
		{16, 400, 200},
		{16, 400, 256},
		{24, 400, 128},
		{24, 400, 200},
		{32, 400, 200},
	}

	fmt.Printf("\n%-6s %-6s  %-10s  %-12s  %s\n", "M", "efSearch", "Recall@10", "Queries/sec", "Notes")
	fmt.Printf("%-6s %-6s  %-10s  %-12s  %s\n", "------", "------", "----------", "------------", "-----")

	for _, cfg := range configs {
		hnswIdx, _ := NewHNSWIndex(dim, cfg.m, cfg.efC, cfg.efS, metric)
		for i, vec := range vectors {
			hnswIdx.Insert(uint64(i), vec)
		}

		var totalRecall float64
		start := testing.Benchmark(func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				query := queries[i%numQueries]
				hnswIdx.Search(ctx, query, k, 0)
			}
		})

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
		avgRecall := totalRecall / float64(numQueries)

		notes := ""
		if avgRecall >= 0.99 {
			notes = "ChromaDB-level"
		}

		fmt.Printf("%-6d %-6d  %-10.4f  %-12.0f  %s\n",
			cfg.m, cfg.efS, avgRecall, float64(start.N)/start.T.Seconds(), notes)
	}
}
