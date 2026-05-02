package quantize

import (
	"context"

	"github.com/veda/vectordb/internal/distance"
	"github.com/veda/vectordb/internal/index"
)

// InnerQuantizedIndex represents a quantized index (e.g. IVF or HNSW) that operates
// on uint8 codes instead of float32 vectors.
type InnerQuantizedIndex interface {
	Insert(id uint64, code []uint8) error
	Search(ctx context.Context, query []uint8, k int, nprobe int) ([]index.SearchResult, error)
	Delete(id uint64) error
	Len() int
	Rebuild() error
	Flush() error
}

// RawVectorSource provides access to raw float32 vectors for re-ranking.
type RawVectorSource interface {
	GetVector(id uint64) ([]float32, error)
}

// QuantizedIndex wraps a quantized index with an exact float32 re-ranking pipeline.
// It implements the standard index.Index interface.
type QuantizedIndex struct {
	dim        int
	cal        *ScalarQuantizer
	quantIndex InnerQuantizedIndex
	rawSource  RawVectorSource
}

// NewQuantizedIndex creates a new quantized index wrapper.
func NewQuantizedIndex(dim int, quantizer *ScalarQuantizer, inner InnerQuantizedIndex, raw RawVectorSource) *QuantizedIndex {
	return &QuantizedIndex{
		dim:        dim,
		cal:        quantizer,
		quantIndex: inner,
		rawSource:  raw,
	}
}

func (idx *QuantizedIndex) Insert(id uint64, vector []float32) error {
	code, err := idx.cal.Encode(vector)
	if err != nil {
		return err
	}
	return idx.quantIndex.Insert(id, code)
}

// Search implements a 10x oversampling on the quantized index, followed by
// an exact re-ranking using the raw float32 vectors.
func (idx *QuantizedIndex) Search(ctx context.Context, query []float32, k, nprobe int) ([]index.SearchResult, error) {
	// Stage 1: approximate search on quantized index — 10× oversampling
	// Fetch k*10 candidates because quantization lowers precision; re-ranking recovers it.
	qQuery, err := idx.cal.Encode(query)
	if err != nil {
		return nil, err
	}

	searchK := k * 10
	candidates, err := idx.quantIndex.Search(ctx, qQuery, searchK, nprobe)
	if err != nil {
		return nil, err
	}

	if len(candidates) == 0 {
		return nil, nil
	}

	// Stage 2: fetch raw float32 vectors for all candidates
	candidateVecs := make([]float32, 0, len(candidates)*idx.dim)
	validCandidates := make([]index.SearchResult, 0, len(candidates))

	for _, c := range candidates {
		rawVec, err := idx.rawSource.GetVector(c.ID)
		if err == nil && len(rawVec) == idx.dim {
			candidateVecs = append(candidateVecs, rawVec...)
			validCandidates = append(validCandidates, c)
		}
	}

	if len(validCandidates) == 0 {
		return nil, nil
	}

	// Stage 3: exact re-rank using float32 distances
	exactDists := make([]float32, len(validCandidates))
	distance.L2Batch(query, candidateVecs, len(validCandidates), idx.dim, exactDists)

	// Update distances to the exact ones
	for i := range validCandidates {
		validCandidates[i].Distance = exactDists[i]
	}

	// Stage 4: top-K from re-ranked candidates
	return topK(validCandidates, k), nil
}

func (idx *QuantizedIndex) Delete(id uint64) error {
	return idx.quantIndex.Delete(id)
}

func (idx *QuantizedIndex) Len() int {
	return idx.quantIndex.Len()
}

func (idx *QuantizedIndex) Rebuild() error {
	return idx.quantIndex.Rebuild()
}

func (idx *QuantizedIndex) Flush() error {
	return idx.quantIndex.Flush()
}

// topK returns the k elements with the smallest distance.
// Uses a simple selection sort for small k.
func topK(results []index.SearchResult, k int) []index.SearchResult {
	if len(results) <= k {
		// Sort the entire slice
		for i := 0; i < len(results); i++ {
			minIdx := i
			for j := i + 1; j < len(results); j++ {
				if results[j].Distance < results[minIdx].Distance {
					minIdx = j
				}
			}
			results[i], results[minIdx] = results[minIdx], results[i]
		}
		return results
	}

	// Find top k
	for i := 0; i < k; i++ {
		minIdx := i
		for j := i + 1; j < len(results); j++ {
			if results[j].Distance < results[minIdx].Distance {
				minIdx = j
			}
		}
		results[i], results[minIdx] = results[minIdx], results[i]
	}

	return results[:k]
}
