// Package bench provides end-to-end benchmarks for cross-cutting VectorDB operations.
//
// Individual component benchmarks (distance, flat, hnsw, ivf, spann, sparse, PQ, SQ,
// cache, rendezvous) live in their respective package _test.go files.
// This file contains system-level benchmarks that exercise multiple layers.
package bench

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/veda/vectordb/internal/collection"
	"github.com/veda/vectordb/internal/metadata"
	"github.com/veda/vectordb/internal/storage"
)

// BenchmarkEndToEnd_InsertSearch benchmarks the full path:
// API → Collection Manager → WAL → Index → Search
func BenchmarkEndToEnd_InsertSearch(b *testing.B) {
	for _, indexType := range []string{"flat", "hnsw"} {
		b.Run(indexType, func(b *testing.B) {
			tmpDir := b.TempDir()

			sysdb, err := metadata.NewSysDB(tmpDir + "/sysdb.sqlite")
			if err != nil {
				b.Fatal(err)
			}
			defer sysdb.Close()

			wal, err := storage.NewSQLiteWAL(tmpDir + "/wal.sqlite")
			if err != nil {
				b.Fatal(err)
			}
			defer wal.Close()

			mgr, err := collection.NewManager(sysdb, wal)
			if err != nil {
				b.Fatal(err)
			}

			dim := 128
			col, err := mgr.CreateCollection("bench-"+indexType, dim, "l2", indexType)
			if err != nil {
				b.Fatal(err)
			}

			// Pre-populate with 1000 vectors
			rng := rand.New(rand.NewSource(42))
			for i := 0; i < 1000; i++ {
				vec := make([]float32, dim)
				for j := range vec {
					vec[j] = rng.Float32()
				}
				if err := mgr.InsertVectors(context.Background(), col.ID,
					[]uint64{uint64(i)}, [][]float32{vec}, nil); err != nil {
					b.Fatal(err)
				}
			}

			// Benchmark: insert + search cycle
			query := make([]float32, dim)
			for j := range query {
				query[j] = rng.Float32()
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				id := uint64(1000 + i)
				vec := make([]float32, dim)
				for j := range vec {
					vec[j] = rng.Float32()
				}
				mgr.InsertVectors(context.Background(), col.ID,
					[]uint64{id}, [][]float32{vec}, nil)
				mgr.SearchVectors(context.Background(), col.ID, query, 10, 5, nil)
			}
		})
	}
}

// BenchmarkEndToEnd_Insert benchmarks raw insert throughput through the full stack.
func BenchmarkEndToEnd_Insert(b *testing.B) {
	tmpDir := b.TempDir()

	sysdb, err := metadata.NewSysDB(tmpDir + "/sysdb.sqlite")
	if err != nil {
		b.Fatal(err)
	}
	defer sysdb.Close()

	wal, err := storage.NewSQLiteWAL(tmpDir + "/wal.sqlite")
	if err != nil {
		b.Fatal(err)
	}
	defer wal.Close()

	mgr, err := collection.NewManager(sysdb, wal)
	if err != nil {
		b.Fatal(err)
	}

	dim := 128
	col, err := mgr.CreateCollection("bench-insert", dim, "l2", "flat")
	if err != nil {
		b.Fatal(err)
	}

	rng := rand.New(rand.NewSource(42))
	vecs := make([][]float32, b.N)
	for i := range vecs {
		vecs[i] = make([]float32, dim)
		for j := range vecs[i] {
			vecs[i][j] = rng.Float32()
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mgr.InsertVectors(context.Background(), col.ID,
			[]uint64{uint64(i)}, [][]float32{vecs[i]}, nil)
	}

	b.ReportMetric(float64(b.N), "vectors")
	b.ReportMetric(float64(dim*4*b.N)/(1024*1024), "MB_indexed")
}

// BenchmarkEndToEnd_Search benchmarks search-only throughput.
func BenchmarkEndToEnd_Search(b *testing.B) {
	for _, tc := range []struct {
		name      string
		indexType string
		n         int
	}{
		{"flat/1K", "flat", 1000},
		{"hnsw/10K", "hnsw", 10000},
	} {
		b.Run(tc.name, func(b *testing.B) {
			tmpDir := b.TempDir()

			sysdb, err := metadata.NewSysDB(tmpDir + "/sysdb.sqlite")
			if err != nil {
				b.Fatal(err)
			}
			defer sysdb.Close()

			wal, err := storage.NewSQLiteWAL(tmpDir + "/wal.sqlite")
			if err != nil {
				b.Fatal(err)
			}
			defer wal.Close()

			mgr, err := collection.NewManager(sysdb, wal)
			if err != nil {
				b.Fatal(err)
			}

			dim := 128
			col, err := mgr.CreateCollection(fmt.Sprintf("bench-%s", tc.name), dim, "l2", tc.indexType)
			if err != nil {
				b.Fatal(err)
			}

			rng := rand.New(rand.NewSource(42))
			for i := 0; i < tc.n; i++ {
				vec := make([]float32, dim)
				for j := range vec {
					vec[j] = rng.Float32()
				}
				mgr.InsertVectors(context.Background(), col.ID,
					[]uint64{uint64(i)}, [][]float32{vec}, nil)
			}

			query := make([]float32, dim)
			for j := range query {
				query[j] = rng.Float32()
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				mgr.SearchVectors(context.Background(), col.ID, query, 10, 5, nil)
			}
		})
	}
}
