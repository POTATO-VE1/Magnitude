// Package quantize implements Scalar Quantization (SQ8) for compressing float32
// vectors into uint8 representations. SQ8 reduces memory usage by 4× with
// minimal recall degradation for most embedding models.
//
// Quantization pipeline:
//  1. During training, compute per-dimension min/max across all vectors.
//  2. Map each float32 component to uint8: q = round((x - min) / (max - min) * 255)
//  3. Store quantized codes + (min, scale) per dimension for reconstruction.
//  4. Distance computation uses reconstructed approximations.
//
// ChromaDB Cloud uses scalar quantization on top of SPANN (QuantizedSPANN).
package quantize

import (
	"fmt"
	"math"
)

// ScalarQuantizer performs per-dimension min-max scalar quantization to uint8.
// After training on a representative set of vectors, it can encode any vector
// of the same dimension into a compact uint8 representation.
type ScalarQuantizer struct {
	Dim    int       // vector dimension
	Mins   []float32 // per-dimension minimum values
	Scales []float32 // per-dimension (max-min)/255 scales
	trained bool
}

// NewScalarQuantizer creates a new untrained scalar quantizer for the given dimension.
func NewScalarQuantizer(dim int) *ScalarQuantizer {
	return &ScalarQuantizer{
		Dim:    dim,
		Mins:   make([]float32, dim),
		Scales: make([]float32, dim),
	}
}

// Train computes per-dimension min/max from a set of training vectors.
// vectors is a flat [n * dim] row-major slice.
func (sq *ScalarQuantizer) Train(vectors []float32, n int) error {
	if n == 0 {
		return fmt.Errorf("quantize: cannot train on zero vectors")
	}
	if len(vectors) != n*sq.Dim {
		return fmt.Errorf("quantize: expected %d values, got %d", n*sq.Dim, len(vectors))
	}

	mins := make([]float32, sq.Dim)
	maxs := make([]float32, sq.Dim)

	// Initialize with first vector
	for d := 0; d < sq.Dim; d++ {
		mins[d] = vectors[d]
		maxs[d] = vectors[d]
	}

	// Scan all vectors
	for i := 1; i < n; i++ {
		offset := i * sq.Dim
		for d := 0; d < sq.Dim; d++ {
			v := vectors[offset+d]
			if v < mins[d] {
				mins[d] = v
			}
			if v > maxs[d] {
				maxs[d] = v
			}
		}
	}

	// Compute scales
	for d := 0; d < sq.Dim; d++ {
		sq.Mins[d] = mins[d]
		rangeD := maxs[d] - mins[d]
		if rangeD == 0 {
			sq.Scales[d] = 0 // constant dimension
		} else {
			sq.Scales[d] = rangeD / 255.0
		}
	}

	sq.trained = true
	return nil
}

// Encode quantizes a float32 vector into a uint8 code.
// The quantizer must be trained first.
func (sq *ScalarQuantizer) Encode(vector []float32) ([]uint8, error) {
	if !sq.trained {
		return nil, fmt.Errorf("quantize: quantizer not trained")
	}
	if len(vector) != sq.Dim {
		return nil, fmt.Errorf("quantize: dimension mismatch: expected %d, got %d", sq.Dim, len(vector))
	}

	code := make([]uint8, sq.Dim)
	for d := 0; d < sq.Dim; d++ {
		if sq.Scales[d] == 0 {
			code[d] = 0
			continue
		}
		val := (vector[d] - sq.Mins[d]) / sq.Scales[d]
		// Clamp to [0, 255]
		if val < 0 {
			val = 0
		} else if val > 255 {
			val = 255
		}
		code[d] = uint8(math.Round(float64(val)))
	}
	return code, nil
}

// Decode reconstructs an approximate float32 vector from a uint8 code.
func (sq *ScalarQuantizer) Decode(code []uint8) ([]float32, error) {
	if !sq.trained {
		return nil, fmt.Errorf("quantize: quantizer not trained")
	}
	if len(code) != sq.Dim {
		return nil, fmt.Errorf("quantize: code length mismatch: expected %d, got %d", sq.Dim, len(code))
	}

	vector := make([]float32, sq.Dim)
	for d := 0; d < sq.Dim; d++ {
		vector[d] = sq.Mins[d] + float32(code[d])*sq.Scales[d]
	}
	return vector, nil
}

// EncodeBatch quantizes multiple vectors. Returns a flat [n * dim] uint8 slice.
func (sq *ScalarQuantizer) EncodeBatch(vectors []float32, n int) ([]uint8, error) {
	if !sq.trained {
		return nil, fmt.Errorf("quantize: quantizer not trained")
	}

	codes := make([]uint8, n*sq.Dim)
	for i := 0; i < n; i++ {
		vec := vectors[i*sq.Dim : (i+1)*sq.Dim]
		for d := 0; d < sq.Dim; d++ {
			if sq.Scales[d] == 0 {
				codes[i*sq.Dim+d] = 0
				continue
			}
			val := (vec[d] - sq.Mins[d]) / sq.Scales[d]
			if val < 0 {
				val = 0
			} else if val > 255 {
				val = 255
			}
			codes[i*sq.Dim+d] = uint8(math.Round(float64(val)))
		}
	}
	return codes, nil
}

// DistanceL2SQ computes approximate L2 squared distance between a float32 query
// and a uint8-encoded vector using the quantizer's reconstruction parameters.
// This avoids full decompression — we reconstruct on-the-fly per dimension.
func (sq *ScalarQuantizer) DistanceL2SQ(query []float32, code []uint8) float32 {
	var sum float32
	for d := 0; d < sq.Dim; d++ {
		reconstructed := sq.Mins[d] + float32(code[d])*sq.Scales[d]
		diff := query[d] - reconstructed
		sum += diff * diff
	}
	return sum
}

// CompressionRatio returns the memory savings: float32 = 4 bytes/dim, uint8 = 1 byte/dim.
func (sq *ScalarQuantizer) CompressionRatio() float64 {
	return 4.0 // 4× compression
}

// IsTrained returns whether the quantizer has been trained.
func (sq *ScalarQuantizer) IsTrained() bool {
	return sq.trained
}
