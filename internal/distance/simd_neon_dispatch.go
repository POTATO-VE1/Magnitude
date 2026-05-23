//go:build arm64 && !noasm

package distance

import (
	"math"

	"github.com/POTATO-VE1/Magnitude/internal/distance/neon"
)

// l2BatchSIMD computes L2 squared distance using NEON intrinsics.
func l2BatchSIMD(query, matrix []float32, n, dim int, results []float32) {
	if n <= 0 || dim <= 0 {
		return
	}
	if n > math.MaxInt32 || dim > math.MaxInt32 {
		L2BatchPure(query, matrix, n, dim, results)
		return
	}
	if len(query) < dim || len(matrix) < n*dim || len(results) < n {
		L2BatchPure(query, matrix, n, dim, results)
		return
	}
	neon.L2BatchNEON(query, matrix, n, dim, results)
}

// dotBatchSIMD computes dot product using NEON intrinsics.
func dotBatchSIMD(query, matrix []float32, n, dim int, results []float32) {
	if n <= 0 || dim <= 0 {
		return
	}
	if n > math.MaxInt32 || dim > math.MaxInt32 {
		DotBatchPure(query, matrix, n, dim, results)
		return
	}
	if len(query) < dim || len(matrix) < n*dim || len(results) < n {
		DotBatchPure(query, matrix, n, dim, results)
		return
	}
	neon.DotBatchNEON(query, matrix, n, dim, results)
}

// cosineBatchSIMD computes cosine distance using NEON intrinsics.
func cosineBatchSIMD(query, matrix []float32, n, dim int, queryNorm float32, results []float32) {
	if n <= 0 || dim <= 0 {
		return
	}
	if n > math.MaxInt32 || dim > math.MaxInt32 {
		CosineBatchPure(query, matrix, n, dim, results)
		return
	}
	if len(query) < dim || len(matrix) < n*dim || len(results) < n {
		CosineBatchPure(query, matrix, n, dim, results)
		return
	}
	neon.CosineBatchNEON(query, matrix, n, dim, queryNorm, results)
}
