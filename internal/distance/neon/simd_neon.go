//go:build arm64
// +build arm64

package neon

/*
#cgo CFLAGS: -O3 -march=armv8-a+simd
#cgo LDFLAGS: -lm
#include "simd_neon.h"
*/
import "C"
import "unsafe"

// L2BatchNEON computes L2 squared distance using NEON intrinsics.
func L2BatchNEON(query, matrix []float32, n, dim int, results []float32) {
	C.l2_batch_neon(
		(*C.float)(unsafe.Pointer(&query[0])),
		(*C.float)(unsafe.Pointer(&matrix[0])),
		C.int(n), C.int(dim),
		(*C.float)(unsafe.Pointer(&results[0])),
	)
}

// DotBatchNEON computes dot product using NEON intrinsics.
func DotBatchNEON(query, matrix []float32, n, dim int, results []float32) {
	C.dot_batch_neon(
		(*C.float)(unsafe.Pointer(&query[0])),
		(*C.float)(unsafe.Pointer(&matrix[0])),
		C.int(n), C.int(dim),
		(*C.float)(unsafe.Pointer(&results[0])),
	)
}

// CosineBatchNEON computes cosine distance using NEON intrinsics.
func CosineBatchNEON(query, matrix []float32, n, dim int, queryNorm float32, results []float32) {
	C.cosine_batch_neon(
		(*C.float)(unsafe.Pointer(&query[0])),
		(*C.float)(unsafe.Pointer(&matrix[0])),
		C.int(n), C.int(dim),
		C.float(queryNorm),
		(*C.float)(unsafe.Pointer(&results[0])),
	)
}
