package distance

import "golang.org/x/sys/cpu"

var useSIMD bool

// simdThreshold is the minimum number of vectors before using CGO SIMD.
// Below this, the ~90ns CGO overhead dominates the computation time.
const simdThreshold = 64

func init() {
	useSIMD = cpu.X86.HasAVX2 && cpu.X86.HasFMA
}

// L2Batch computes L2 squared distance between query and n vectors.
// Uses AVX2 SIMD on supporting CPUs for large batches, optimized pure Go otherwise.
func L2Batch(query, matrix []float32, n, dim int, results []float32) {
	if useSIMD && n >= simdThreshold {
		l2BatchSIMD(query, matrix, n, dim, results)
	} else {
		L2BatchPure(query, matrix, n, dim, results)
	}
}

// DotBatch computes dot product similarity of query with n vectors.
func DotBatch(query, matrix []float32, n, dim int, results []float32) {
	if useSIMD && n >= simdThreshold {
		dotBatchSIMD(query, matrix, n, dim, results)
	} else {
		DotBatchPure(query, matrix, n, dim, results)
	}
}

// CosineBatch computes cosine distance from query to n vectors.
func CosineBatch(query, matrix []float32, n, dim int, results []float32) {
	if useSIMD && n >= simdThreshold {
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
