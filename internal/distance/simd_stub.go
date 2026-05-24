//go:build (!amd64 && !arm64) || (arm64 && !cgo)
// +build !amd64,!arm64

package distance

func l2BatchSIMD(query, matrix []float32, n, dim int, results []float32) {
	panic("l2BatchSIMD called without SIMD support")
}

func dotBatchSIMD(query, matrix []float32, n, dim int, results []float32) {
	panic("dotBatchSIMD called without SIMD support")
}

func cosineBatchSIMD(query, matrix []float32, n, dim int, queryNorm float32, results []float32) {
	panic("cosineBatchSIMD called without SIMD support")
}
