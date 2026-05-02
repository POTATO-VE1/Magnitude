// Package search provides search fusion algorithms for hybrid search.
//
// RRF (Reciprocal Rank Fusion) merges multiple ranked result lists without
// requiring score normalization. From the 2009 Cormack et al. paper:
//
//	RRF_score(d) = Σ 1 / (k + rank_i(d))
//
// where k is a constant (default 60) and rank_i(d) is the 1-indexed rank
// of document d in the i-th result list. Documents not present in a list
// receive no contribution from that list.
//
// RRF is superior to score-based fusion because:
//   - Ranks are dimensionless (no normalization needed)
//   - Robust to score scale differences between dense and sparse search
//   - Simple, parameter-free (k=60 works across nearly all datasets)
package search

import (
	"sort"
)

// RankedResult represents a result with an ID and a score.
type RankedResult struct {
	ID    uint64
	Score float32
}

// RRF merges multiple ranked result lists using Reciprocal Rank Fusion.
// k=60 is the industry standard from the original 2009 paper.
// Higher k → less sensitive to rank differences between lists.
// Returns results sorted by fused score descending.
func RRF(lists [][]RankedResult, k float64) []RankedResult {
	if k <= 0 {
		k = 60
	}

	scores := make(map[uint64]float64)

	for _, list := range lists {
		for rank, result := range list {
			// RRF: 1 / (k + rank)  [rank is 0-indexed here, paper uses 1-indexed]
			scores[result.ID] += 1.0 / (k + float64(rank+1))
		}
	}

	merged := make([]RankedResult, 0, len(scores))
	for id, score := range scores {
		merged = append(merged, RankedResult{ID: id, Score: float32(score)})
	}

	sort.Slice(merged, func(i, j int) bool {
		return merged[i].Score > merged[j].Score
	})

	return merged
}

// RRFTopK merges multiple ranked lists and returns only the top-K results.
func RRFTopK(lists [][]RankedResult, k float64, topK int) []RankedResult {
	merged := RRF(lists, k)
	if len(merged) > topK {
		return merged[:topK]
	}
	return merged
}

// WeightedRRF applies per-list weight multipliers before fusion.
// weights[i] scales the RRF contribution of lists[i].
// Use alpha=1.0 for dense, alpha=0.5 for sparse to favor semantic search.
func WeightedRRF(lists [][]RankedResult, weights []float64, k float64) []RankedResult {
	if k <= 0 {
		k = 60
	}
	if len(weights) != len(lists) {
		// Equal weights if not specified
		weights = make([]float64, len(lists))
		for i := range weights {
			weights[i] = 1.0
		}
	}

	scores := make(map[uint64]float64)
	for i, list := range lists {
		w := weights[i]
		for rank, result := range list {
			scores[result.ID] += w / (k + float64(rank+1))
		}
	}

	merged := make([]RankedResult, 0, len(scores))
	for id, score := range scores {
		merged = append(merged, RankedResult{ID: id, Score: float32(score)})
	}

	sort.Slice(merged, func(i, j int) bool {
		return merged[i].Score > merged[j].Score
	})

	return merged
}
