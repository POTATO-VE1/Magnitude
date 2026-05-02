//go:build !amd64
// +build !amd64

package distance

// L2BatchSIMD is a stub for non-amd64 architectures.
// It will never be called because cpu.X86.HasAVX2 will be false.
func L2BatchSIMD(query, matrix []float32, n, dim int, results []float32) {
	panic("L2BatchSIMD called on non-amd64 architecture")
}
