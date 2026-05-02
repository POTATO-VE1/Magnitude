package index

// RecallAtK computes the fraction of ground-truth results found in approximate results.
// This is the single most important quality metric for any ANN index.
//
// truth: the correct Top-K results from a brute-force (exact) search.
// approx: the Top-K results from an approximate index (IVF, HNSW, SPANN).
//
// Returns a value in [0.0, 1.0]:
//   - 1.0 = perfect recall (all exact results found)
//   - 0.0 = no overlap at all
//
// Usage: run RecallAtK after every index change as a regression test.
// If recall drops below 0.85, something is wrong with the index.
func RecallAtK(truth, approx []SearchResult) float64 {
	if len(truth) == 0 {
		return 1.0 // vacuously true
	}

	truthSet := make(map[uint64]struct{}, len(truth))
	for _, r := range truth {
		truthSet[r.ID] = struct{}{}
	}

	var hits int
	for _, r := range approx {
		if _, ok := truthSet[r.ID]; ok {
			hits++
		}
	}

	return float64(hits) / float64(len(truth))
}
