//go:build amd64
// +build amd64

package distance

/*
#cgo CFLAGS: -O3 -mavx2 -mfma
#cgo LDFLAGS: -lm
#include "simd_avx2.h"
*/
import "C"
import "unsafe"

// l2BatchSIMD computes L2 squared distance using AVX2 intrinsics via CGO.
func l2BatchSIMD(query, matrix []float32, n, dim int, results []float32) {
	if len(query) < dim || len(matrix) < n*dim || len(results) < n {
		// Bounds check failed — fall back to pure Go to avoid segfault
		L2BatchPure(query, matrix, n, dim, results)
		return
	}
	C.l2_batch_avx2(
		(*C.float)(unsafe.Pointer(&query[0])),
		(*C.float)(unsafe.Pointer(&matrix[0])),
		C.int(n), C.int(dim),
		(*C.float)(unsafe.Pointer(&results[0])),
	)
}

// dotBatchSIMD computes dot product using AVX2 intrinsics via CGO.
func dotBatchSIMD(query, matrix []float32, n, dim int, results []float32) {
	if len(query) < dim || len(matrix) < n*dim || len(results) < n {
		DotBatchPure(query, matrix, n, dim, results)
		return
	}
	C.dot_batch_avx2(
		(*C.float)(unsafe.Pointer(&query[0])),
		(*C.float)(unsafe.Pointer(&matrix[0])),
		C.int(n), C.int(dim),
		(*C.float)(unsafe.Pointer(&results[0])),
	)
}

// cosineBatchSIMD computes cosine distance using AVX2 intrinsics via CGO.
func cosineBatchSIMD(query, matrix []float32, n, dim int, queryNorm float32, results []float32) {
	if len(query) < dim || len(matrix) < n*dim || len(results) < n {
		CosineBatchPure(query, matrix, n, dim, results)
		return
	}
	C.cosine_batch_avx2(
		(*C.float)(unsafe.Pointer(&query[0])),
		(*C.float)(unsafe.Pointer(&matrix[0])),
		C.int(n), C.int(dim),
		C.float(queryNorm),
		(*C.float)(unsafe.Pointer(&results[0])),
	)
}
