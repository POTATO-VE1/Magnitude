package distance

// BatchDistance computes distance from query to n vectors in matrix.
// Results are written to pre-allocated results slice (len >= n).
// Dispatches to the fastest available implementation based on metric.
//
// Supported metrics: "l2", "cosine", "dot", "manhattan".
func BatchDistance(query, matrix []float32, n, dim int, metric string, results []float32) {
	switch metric {
	case "l2":
		L2Batch(query, matrix, n, dim, results)
	case "cosine":
		CosineBatch(query, matrix, n, dim, results)
	case "dot":
		DotBatch(query, matrix, n, dim, results)
	case "manhattan":
		ManhattanBatch(query, matrix, n, dim, results)
	default:
		for i := range results[:n] {
			results[i] = 0
		}
	}
}
