// Package quantize — Product Quantization (PQ).
// PQ splits the D-dimensional vector space into M sub-spaces of D/M dimensions each.
// Each sub-space is independently quantized with a codebook of K=256 centroids (uint8 code).
// Total storage: M bytes per vector (vs 4*D bytes for float32).
//
// PQ enables compressed-domain distance computation (ADC — Asymmetric Distance Computation):
//  1. For each sub-space m, compute distances from query sub-vector to all 256 centroids.
//  2. Look up precomputed distance tables to accumulate approximate full distance.
//  3. No vector decompression needed — ADC is faster than exact L2 on many datasets.
package quantize

import (
	"fmt"
	"math"
	"math/rand"
)

// ProductQuantizer implements Product Quantization with K-Means codebook training.
type ProductQuantizer struct {
	Dim       int       // full vector dimension
	M         int       // number of sub-spaces
	SubDim    int       // dimension per sub-space (Dim / M)
	KBook     int       // codebook size per sub-space (always 256)
	Codebooks []float32 // shape [M * KBook * SubDim], row-major
	trained   bool
}

// NewProductQuantizer creates a new untrained PQ with M sub-spaces.
// Dim must be divisible by M.
func NewProductQuantizer(dim, m int) (*ProductQuantizer, error) {
	if dim <= 0 || m <= 0 {
		return nil, fmt.Errorf("quantize: dim and m must be > 0")
	}
	if dim%m != 0 {
		return nil, fmt.Errorf("quantize: dim (%d) must be divisible by m (%d)", dim, m)
	}
	return &ProductQuantizer{
		Dim:    dim,
		M:      m,
		SubDim: dim / m,
		KBook:  256,
	}, nil
}

// Train builds codebooks using K-Means on training vectors.
// vectors is a flat [n * dim] row-major slice.
func (pq *ProductQuantizer) Train(vectors []float32, n int, maxIter int) error {
	if n == 0 {
		return fmt.Errorf("quantize: cannot train on zero vectors")
	}
	if maxIter <= 0 {
		maxIter = 25
	}

	rng := rand.New(rand.NewSource(42))
	pq.Codebooks = make([]float32, pq.M*pq.KBook*pq.SubDim)

	// Train each sub-space independently
	for m := 0; m < pq.M; m++ {
		// Extract sub-vectors for this sub-space
		subVecs := make([]float32, n*pq.SubDim)
		for i := 0; i < n; i++ {
			srcOffset := i*pq.Dim + m*pq.SubDim
			dstOffset := i * pq.SubDim
			copy(subVecs[dstOffset:dstOffset+pq.SubDim], vectors[srcOffset:srcOffset+pq.SubDim])
		}

		// Run K-Means on this sub-space
		effectiveK := pq.KBook
		if n < effectiveK {
			effectiveK = n
		}

		centroids := pq.kmeansSubspace(subVecs, n, pq.SubDim, effectiveK, maxIter, rng)

		// Copy centroids into codebook
		cbOffset := m * pq.KBook * pq.SubDim
		copy(pq.Codebooks[cbOffset:cbOffset+effectiveK*pq.SubDim], centroids)

		// Zero-pad remaining centroids if n < 256
		for k := effectiveK; k < pq.KBook; k++ {
			for d := 0; d < pq.SubDim; d++ {
				pq.Codebooks[cbOffset+k*pq.SubDim+d] = 0
			}
		}
	}

	pq.trained = true
	return nil
}

// kmeansSubspace runs simple K-Means on sub-vectors.
func (pq *ProductQuantizer) kmeansSubspace(subVecs []float32, n, subDim, k, maxIter int, rng *rand.Rand) []float32 {
	centroids := make([]float32, k*subDim)

	// Initialize: random selection
	usedIndices := make(map[int]bool)
	for c := 0; c < k; c++ {
		idx := rng.Intn(n)
		for usedIndices[idx] {
			idx = rng.Intn(n)
		}
		usedIndices[idx] = true
		copy(centroids[c*subDim:(c+1)*subDim], subVecs[idx*subDim:(idx+1)*subDim])
	}

	assignments := make([]int, n)

	for iter := 0; iter < maxIter; iter++ {
		// Assignment step
		changes := 0
		for i := 0; i < n; i++ {
			vec := subVecs[i*subDim : (i+1)*subDim]
			nearest := 0
			nearestDist := float32(math.MaxFloat32)
			for c := 0; c < k; c++ {
				centroid := centroids[c*subDim : (c+1)*subDim]
				var d float32
				for d2 := 0; d2 < subDim; d2++ {
					diff := vec[d2] - centroid[d2]
					d += diff * diff
				}
				if d < nearestDist {
					nearestDist = d
					nearest = c
				}
			}
			if nearest != assignments[i] {
				assignments[i] = nearest
				changes++
			}
		}

		if changes == 0 {
			break
		}

		// Update step
		newCentroids := make([]float32, k*subDim)
		counts := make([]int, k)
		for i := 0; i < n; i++ {
			c := assignments[i]
			counts[c]++
			src := subVecs[i*subDim : (i+1)*subDim]
			dst := newCentroids[c*subDim : (c+1)*subDim]
			for d := 0; d < subDim; d++ {
				dst[d] += src[d]
			}
		}
		for c := 0; c < k; c++ {
			if counts[c] > 0 {
				centroid := newCentroids[c*subDim : (c+1)*subDim]
				invN := 1.0 / float32(counts[c])
				for d := range centroid {
					centroid[d] *= invN
				}
			}
		}
		centroids = newCentroids
	}

	return centroids
}

// Encode quantizes a float32 vector into M uint8 codes.
func (pq *ProductQuantizer) Encode(vector []float32) ([]uint8, error) {
	if !pq.trained {
		return nil, fmt.Errorf("quantize: PQ not trained")
	}
	if len(vector) != pq.Dim {
		return nil, fmt.Errorf("quantize: dimension mismatch: expected %d, got %d", pq.Dim, len(vector))
	}

	codes := make([]uint8, pq.M)
	for m := 0; m < pq.M; m++ {
		subVec := vector[m*pq.SubDim : (m+1)*pq.SubDim]
		cbOffset := m * pq.KBook * pq.SubDim

		nearest := 0
		nearestDist := float32(math.MaxFloat32)
		for k := 0; k < pq.KBook; k++ {
			centroid := pq.Codebooks[cbOffset+k*pq.SubDim : cbOffset+(k+1)*pq.SubDim]
			var d float32
			for i := 0; i < pq.SubDim; i++ {
				diff := subVec[i] - centroid[i]
				d += diff * diff
			}
			if d < nearestDist {
				nearestDist = d
				nearest = k
			}
		}
		codes[m] = uint8(nearest)
	}
	return codes, nil
}

// Decode reconstructs an approximate vector from PQ codes.
func (pq *ProductQuantizer) Decode(codes []uint8) ([]float32, error) {
	if !pq.trained {
		return nil, fmt.Errorf("quantize: PQ not trained")
	}
	if len(codes) != pq.M {
		return nil, fmt.Errorf("quantize: code length mismatch: expected %d, got %d", pq.M, len(codes))
	}

	vector := make([]float32, pq.Dim)
	for m := 0; m < pq.M; m++ {
		k := int(codes[m])
		cbOffset := m*pq.KBook*pq.SubDim + k*pq.SubDim
		copy(vector[m*pq.SubDim:(m+1)*pq.SubDim], pq.Codebooks[cbOffset:cbOffset+pq.SubDim])
	}
	return vector, nil
}

// BuildDistanceTable precomputes distances from each query sub-vector to all 256
// codebook centroids. This enables ADC (Asymmetric Distance Computation).
// Returns a [M × 256] lookup table.
func (pq *ProductQuantizer) BuildDistanceTable(query []float32) []float32 {
	table := make([]float32, pq.M*pq.KBook)
	for m := 0; m < pq.M; m++ {
		subQuery := query[m*pq.SubDim : (m+1)*pq.SubDim]
		cbOffset := m * pq.KBook * pq.SubDim
		for k := 0; k < pq.KBook; k++ {
			centroid := pq.Codebooks[cbOffset+k*pq.SubDim : cbOffset+(k+1)*pq.SubDim]
			var d float32
			for i := 0; i < pq.SubDim; i++ {
				diff := subQuery[i] - centroid[i]
				d += diff * diff
			}
			table[m*pq.KBook+k] = d
		}
	}
	return table
}

// DistanceADC computes approximate L2 squared distance using the precomputed
// distance table (ADC). This is a pure table lookup — no FP math.
//   cost: M lookups + M additions ≈ O(M) vs O(D) for exact
func (pq *ProductQuantizer) DistanceADC(table []float32, codes []uint8) float32 {
	var sum float32
	for m := 0; m < pq.M; m++ {
		sum += table[m*pq.KBook+int(codes[m])]
	}
	return sum
}

// CompressionRatio returns the memory savings.
// float32 = 4*D bytes, PQ = M bytes → ratio = 4*D/M.
func (pq *ProductQuantizer) CompressionRatio() float64 {
	return float64(4*pq.Dim) / float64(pq.M)
}

// IsTrained returns whether the PQ has been trained.
func (pq *ProductQuantizer) IsTrained() bool {
	return pq.trained
}
