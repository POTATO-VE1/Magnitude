package distance

import "golang.org/x/sys/cpu"

var useSIMD bool

func init() {
	useSIMD = cpu.X86.HasAVX2 && cpu.X86.HasFMA
}

// L2Batch computes L2 squared distance between query and n vectors.
// Uses AVX2 SIMD on supporting CPUs, falls back to optimized pure Go.
func L2Batch(query, matrix []float32, n, dim int, results []float32) {
	if useSIMD {
		l2BatchSIMD(query, matrix, n, dim, results)
	} else {
		L2BatchPure(query, matrix, n, dim, results)
	}
}

// DotBatch computes dot product similarity of query with n vectors.
func DotBatch(query, matrix []float32, n, dim int, results []float32) {
	if useSIMD {
		dotBatchSIMD(query, matrix, n, dim, results)
	} else {
		DotBatchPure(query, matrix, n, dim, results)
	}
}

// CosineBatch computes cosine distance from query to n vectors.
func CosineBatch(query, matrix []float32, n, dim int, results []float32) {
	if useSIMD {
		var queryNorm float32
		for j := 0; j < dim; j++ {
			queryNorm += query[j] * query[j]
		}
		cosineBatchSIMD(query, matrix, n, dim, queryNorm, results)
	} else {
		CosineBatchPure(query, matrix, n, dim, results)
	}
}

// ManhattanBatch computes Manhattan (L1) distance from query to n vectors.
func ManhattanBatch(query, matrix []float32, n, dim int, results []float32) {
	ManhattanBatchPure(query, matrix, n, dim, results)
}
