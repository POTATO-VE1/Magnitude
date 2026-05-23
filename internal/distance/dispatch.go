package distance

// L2Batch computes L2 squared distance between query and n vectors.
func L2Batch(query, matrix []float32, n, dim int, results []float32) {
	L2BatchPure(query, matrix, n, dim, results)
}

// DotBatch computes dot product similarity of query with n vectors.
func DotBatch(query, matrix []float32, n, dim int, results []float32) {
	DotBatchPure(query, matrix, n, dim, results)
}

// CosineBatch computes cosine distance from query to n vectors.
func CosineBatch(query, matrix []float32, n, dim int, results []float32) {
	CosineBatchPure(query, matrix, n, dim, results)
}

// ManhattanBatch computes Manhattan (L1) distance from query to n vectors.
func ManhattanBatch(query, matrix []float32, n, dim int, results []float32) {
	ManhattanBatchPure(query, matrix, n, dim, results)
}
