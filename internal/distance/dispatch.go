package distance

import "golang.org/x/sys/cpu"

var useAVX2 bool

func init() {
	// Only enable AVX2 path if both AVX2 and FMA are supported
	useAVX2 = cpu.X86.HasAVX2 && cpu.X86.HasFMA
}

// L2Batch computes L2 squared distance between query and n vectors in matrix.
// It uses AVX2 SIMD instructions if supported by the CPU.
func L2Batch(query, matrix []float32, n, dim int, results []float32) {
	if useAVX2 {
		L2BatchSIMD(query, matrix, n, dim, results) // CGO AVX2 path
	} else {
		L2BatchPure(query, matrix, n, dim, results) // pure Go fallback
	}
}
