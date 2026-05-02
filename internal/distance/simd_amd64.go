//go:build amd64
// +build amd64

package distance

/*
#cgo CFLAGS: -O3 -mavx2
void l2_batch_avx2(const float* query, const float* matrix, int n, int dim, float* results);
*/
import "C"
import "unsafe"

// L2BatchSIMD computes L2 squared distance in batch using AVX2.
func L2BatchSIMD(query, matrix []float32, n, dim int, results []float32) {
	C.l2_batch_avx2(
		(*C.float)(unsafe.Pointer(&query[0])),
		(*C.float)(unsafe.Pointer(&matrix[0])),
		C.int(n), C.int(dim),
		(*C.float)(unsafe.Pointer(&results[0])),
	)
}
