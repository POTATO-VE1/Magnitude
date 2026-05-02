package search

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRRF_SingleList(t *testing.T) {
	list := []RankedResult{
		{ID: 1, Score: 0.9},
		{ID: 2, Score: 0.8},
		{ID: 3, Score: 0.7},
	}

	result := RRF([][]RankedResult{list}, 60)
	require.Len(t, result, 3)
	// Order should be preserved
	assert.Equal(t, uint64(1), result[0].ID)
	assert.Equal(t, uint64(2), result[1].ID)
	assert.Equal(t, uint64(3), result[2].ID)
}

func TestRRF_TwoLists_Fusion(t *testing.T) {
	// From the roadmap example:
	// Dense: [A=1, C=2, B=3, D=4]
	// Sparse: [B=1, A=2, D=3, C=4]
	dense := []RankedResult{
		{ID: 1}, // A rank 1
		{ID: 3}, // C rank 2
		{ID: 2}, // B rank 3
		{ID: 4}, // D rank 4
	}
	sparse := []RankedResult{
		{ID: 2}, // B rank 1
		{ID: 1}, // A rank 2
		{ID: 4}, // D rank 3
		{ID: 3}, // C rank 4
	}

	result := RRF([][]RankedResult{dense, sparse}, 60)
	require.Len(t, result, 4)

	// With k=60:
	// A: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
	// B: 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226
	// C: 1/(60+2) + 1/(60+4) = 0.01613 + 0.01563 = 0.03176
	// D: 1/(60+4) + 1/(60+3) = 0.01563 + 0.01587 = 0.03150
	// Expected order: A, B, C, D
	assert.Equal(t, uint64(1), result[0].ID, "A should be first")
	assert.Equal(t, uint64(2), result[1].ID, "B should be second")
	assert.Equal(t, uint64(3), result[2].ID, "C should be third")
	assert.Equal(t, uint64(4), result[3].ID, "D should be fourth")
}

func TestRRF_DisjointLists(t *testing.T) {
	list1 := []RankedResult{{ID: 1}, {ID: 2}}
	list2 := []RankedResult{{ID: 3}, {ID: 4}}

	result := RRF([][]RankedResult{list1, list2}, 60)
	assert.Len(t, result, 4)

	// All items at same rank get same score from their respective lists
	// Items at rank 1 get 1/(60+1) each, items at rank 2 get 1/(60+2) each
	assert.InDelta(t, result[0].Score, result[1].Score, 0.001, "rank-1 items should have equal scores")
}

func TestRRF_EmptyLists(t *testing.T) {
	result := RRF(nil, 60)
	assert.Empty(t, result)

	result = RRF([][]RankedResult{}, 60)
	assert.Empty(t, result)

	result = RRF([][]RankedResult{nil, nil}, 60)
	assert.Empty(t, result)
}

func TestRRFTopK(t *testing.T) {
	list := []RankedResult{
		{ID: 1}, {ID: 2}, {ID: 3}, {ID: 4}, {ID: 5},
	}

	result := RRFTopK([][]RankedResult{list}, 60, 3)
	assert.Len(t, result, 3)
	assert.Equal(t, uint64(1), result[0].ID)
}

func TestWeightedRRF(t *testing.T) {
	dense := []RankedResult{{ID: 1}, {ID: 2}} // weighted 2.0
	sparse := []RankedResult{{ID: 2}, {ID: 1}} // weighted 1.0

	// With weights [2.0, 1.0]:
	// ID 1: 2.0/(60+1) + 1.0/(60+2) = 0.03279 + 0.01613 = 0.04892
	// ID 2: 2.0/(60+2) + 1.0/(60+1) = 0.03226 + 0.01639 = 0.04865
	result := WeightedRRF(
		[][]RankedResult{dense, sparse},
		[]float64{2.0, 1.0},
		60,
	)
	require.Len(t, result, 2)
	// ID 1 should rank higher due to higher weight on its rank-1 position
	assert.Equal(t, uint64(1), result[0].ID)
}

func TestWeightedRRF_EqualWeights(t *testing.T) {
	list1 := []RankedResult{{ID: 1}, {ID: 2}}
	list2 := []RankedResult{{ID: 2}, {ID: 1}}

	// Equal weights should give same result as standard RRF
	weighted := WeightedRRF([][]RankedResult{list1, list2}, []float64{1.0, 1.0}, 60)
	standard := RRF([][]RankedResult{list1, list2}, 60)

	require.Len(t, weighted, len(standard))
	for i := range weighted {
		assert.Equal(t, standard[i].ID, weighted[i].ID)
		assert.InDelta(t, standard[i].Score, weighted[i].Score, 1e-6)
	}
}

func TestRRF_DefaultK(t *testing.T) {
	list := []RankedResult{{ID: 1}}
	result := RRF([][]RankedResult{list}, 0) // k=0 → default to 60
	require.Len(t, result, 1)
	expected := float32(1.0 / (60 + 1))
	assert.InDelta(t, expected, result[0].Score, 1e-6)
}
