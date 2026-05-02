package distance

// L2BatchPure is the pure Go fallback for L2 distance batch computation.
func L2BatchPure(query, matrix []float32, n, dim int, results []float32) {
	for i := 0; i < n; i++ {
		row := matrix[i*dim : (i+1)*dim]
		var sum float32
		for j := 0; j < dim; j++ {
			diff := query[j] - row[j]
			sum += diff * diff
		}
		results[i] = sum
	}
}
