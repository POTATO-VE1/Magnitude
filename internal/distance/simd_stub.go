//go:build !amd64 || !cgo
// +build !amd64 !cgo

package distance

const simdEnabled = false

// L2BatchSIMD is a stub for non-amd64 architectures or when CGO is disabled.
// It will never be called because simdEnabled will be false.
func L2BatchSIMD(query, matrix []float32, n, dim int, results []float32) {
	panic("L2BatchSIMD called without SIMD support")
}
