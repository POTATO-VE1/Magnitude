package bench

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/POTATO-VE1/Magnitude/internal/distance"
	"github.com/POTATO-VE1/Magnitude/internal/index/flat"
	"github.com/POTATO-VE1/Magnitude/internal/index/hnsw"
	"github.com/POTATO-VE1/Magnitude/internal/index/ivf"
)

// ── Helpers ─────────────────────────────────────────────────────────────────

func generateVectors(n, dim int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	vectors := make([][]float32, n)
	for i := range vectors {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		vectors[i] = vec
	}
	return vectors
}

func generateQuery(dim int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	vec := make([]float32, dim)
	for j := range vec {
		vec[j] = rng.Float32()
	}
	return vec
}

// ── Flat Index Benchmarks ───────────────────────────────────────────────────

func BenchmarkFlatSearch_128d_1K(b *testing.B) {
	benchFlatSearch(b, 128, 1000, 10)
}

func BenchmarkFlatSearch_128d_10K(b *testing.B) {
	benchFlatSearch(b, 128, 10000, 10)
}

func BenchmarkFlatSearch_768d_1K(b *testing.B) {
	benchFlatSearch(b, 768, 1000, 10)
}

func BenchmarkFlatSearch_768d_10K(b *testing.B) {
	benchFlatSearch(b, 768, 10000, 10)
}

func benchFlatSearch(b *testing.B, dim, n, k int) {
	b.Helper()
	idx, _ := flat.NewFlatIndex(dim, "l2")
	vectors := generateVectors(n, dim, 42)
	for i, vec := range vectors {
		idx.Insert(uint64(i), vec)
	}
	query := generateQuery(dim, 99)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(ctx, query, k, 0)
	}
}

// ── HNSW Index Benchmarks ───────────────────────────────────────────────────

func BenchmarkHNSWSearch_128d_1K(b *testing.B) {
	benchHNSWSearch(b, 128, 1000, 10)
}

func BenchmarkHNSWSearch_128d_10K(b *testing.B) {
	benchHNSWSearch(b, 128, 10000, 10)
}

func BenchmarkHNSWSearch_768d_1K(b *testing.B) {
	benchHNSWSearch(b, 768, 1000, 10)
}

func benchHNSWSearch(b *testing.B, dim, n, k int) {
	b.Helper()
	idx, _ := hnsw.NewHNSWIndex(dim, 16, 200, 50, "l2")
	vectors := generateVectors(n, dim, 42)
	for i, vec := range vectors {
		idx.Insert(uint64(i), vec)
	}
	query := generateQuery(dim, 99)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(ctx, query, k, 0)
	}
}

// ── IVF Index Benchmarks ────────────────────────────────────────────────────

func BenchmarkIVFSearch_128d_1K(b *testing.B) {
	benchIVFSearch(b, 128, 1000, 10)
}

func BenchmarkIVFSearch_128d_10K(b *testing.B) {
	benchIVFSearch(b, 128, 10000, 10)
}

func benchIVFSearch(b *testing.B, dim, n, k int) {
	b.Helper()
	idx, _ := ivf.NewIVFIndex(dim, 64, 8, "l2", 0, 0.10)
	vectors := generateVectors(n, dim, 42)
	for i, vec := range vectors {
		idx.Insert(uint64(i), vec)
	}
	idx.Rebuild()
	query := generateQuery(dim, 99)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(ctx, query, k, 0)
	}
}

// ── Insert Throughput ───────────────────────────────────────────────────────

func BenchmarkInsertThroughput_128d(b *testing.B) {
	benchInsert(b, 128, 10000)
}

func BenchmarkInsertThroughput_768d(b *testing.B) {
	benchInsert(b, 768, 10000)
}

func benchInsert(b *testing.B, dim, n int) {
	b.Helper()
	vectors := generateVectors(n, dim, 42)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx, _ := flat.NewFlatIndex(dim, "l2")
		for j, vec := range vectors {
			idx.Insert(uint64(j), vec)
		}
	}
	b.ReportMetric(float64(n), "vectors/op")
}

// ── Distance Computation ────────────────────────────────────────────────────

func BenchmarkDistanceBatch_128d_1K(b *testing.B) {
	benchDistance(b, 128, 1000)
}

func BenchmarkDistanceBatch_768d_1K(b *testing.B) {
	benchDistance(b, 768, 1000)
}

func benchDistance(b *testing.B, dim, n int) {
	b.Helper()
	query := generateQuery(dim, 42)
	matrix := make([]float32, n*dim)
	for i := range matrix {
		matrix[i] = float32(i)
	}
	results := make([]float32, n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		distance.L2Batch(query, matrix, n, dim, results)
	}
	b.ReportMetric(float64(n*dim), "floats/op")
}

// ── Recall Benchmark ────────────────────────────────────────────────────────

func BenchmarkRecall_128d_1K(b *testing.B) {
	dim := 128
	n := 1000
	k := 10

	flatIdx, _ := flat.NewFlatIndex(dim, "l2")
	hnswIdx, _ := hnsw.NewHNSWIndex(dim, 16, 200, 50, "l2")
	vectors := generateVectors(n, dim, 42)

	for i, vec := range vectors {
		flatIdx.Insert(uint64(i), vec)
		hnswIdx.Insert(uint64(i), vec)
	}

	query := generateQuery(dim, 99)
	ctx := context.Background()

	flatResults, _ := flatIdx.Search(ctx, query, k, 0)
	hnswResults, _ := hnswIdx.Search(ctx, query, k, 0)

	matches := 0
	for _, hr := range hnswResults {
		for _, fr := range flatResults {
			if hr.ID == fr.ID {
				matches++
				break
			}
		}
	}
	recall := float64(matches) / float64(k)
	b.ReportMetric(recall, "recall@10")

	// Benchmark the search itself
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hnswIdx.Search(ctx, query, k, 0)
	}
}

// ── Summary ─────────────────────────────────────────────────────────────────

func BenchmarkSummary(b *testing.B) {
	configs := []struct {
		name string
		dim  int
		n    int
	}{
		{"128d-1K", 128, 1000},
		{"128d-10K", 128, 10000},
		{"768d-1K", 768, 1000},
	}

	for _, cfg := range configs {
		b.Run(fmt.Sprintf("flat/%s", cfg.name), func(b *testing.B) {
			benchFlatSearch(b, cfg.dim, cfg.n, 10)
		})
		b.Run(fmt.Sprintf("hnsw/%s", cfg.name), func(b *testing.B) {
			benchHNSWSearch(b, cfg.dim, cfg.n, 10)
		})
	}
}
