package quantize

import (
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSQ_TrainAndEncode(t *testing.T) {
	dim := 4
	sq := NewScalarQuantizer(dim)

	// Training data with known range
	vectors := []float32{
		0, 0, 0, 0,       // min in all dims
		1, 1, 1, 1,       // max in all dims
		0.5, 0.5, 0.5, 0.5,
	}
	require.NoError(t, sq.Train(vectors, 3))
	assert.True(t, sq.IsTrained())

	// Encode the min vector → should all be 0
	code, err := sq.Encode([]float32{0, 0, 0, 0})
	require.NoError(t, err)
	for _, c := range code {
		assert.Equal(t, uint8(0), c)
	}

	// Encode the max vector → should all be 255
	code, err = sq.Encode([]float32{1, 1, 1, 1})
	require.NoError(t, err)
	for _, c := range code {
		assert.Equal(t, uint8(255), c)
	}

	// Encode midpoint → should be ~128
	code, err = sq.Encode([]float32{0.5, 0.5, 0.5, 0.5})
	require.NoError(t, err)
	for _, c := range code {
		assert.InDelta(t, 128, int(c), 1)
	}
}

func TestSQ_Decode(t *testing.T) {
	dim := 4
	sq := NewScalarQuantizer(dim)
	vectors := []float32{0, 0, 0, 0, 1, 1, 1, 1}
	require.NoError(t, sq.Train(vectors, 2))

	// Encode then decode → should approximate the original
	original := []float32{0.3, 0.7, 0.1, 0.9}
	code, _ := sq.Encode(original)
	decoded, err := sq.Decode(code)
	require.NoError(t, err)

	for i := range original {
		assert.InDelta(t, original[i], decoded[i], 0.01, "dimension %d", i)
	}
}

func TestSQ_DistanceL2SQ(t *testing.T) {
	dim := 4
	sq := NewScalarQuantizer(dim)
	vectors := []float32{0, 0, 0, 0, 1, 1, 1, 1}
	require.NoError(t, sq.Train(vectors, 2))

	vec := []float32{0.5, 0.5, 0.5, 0.5}
	code, _ := sq.Encode(vec)

	query := []float32{0.5, 0.5, 0.5, 0.5}
	dist := sq.DistanceL2SQ(query, code)

	// Self-distance should be very close to 0
	assert.InDelta(t, 0.0, dist, 0.01)
}

func TestSQ_CompressionRatio(t *testing.T) {
	sq := NewScalarQuantizer(128)
	assert.Equal(t, 4.0, sq.CompressionRatio())
}

func TestSQ_EncodeBatch(t *testing.T) {
	dim := 4
	sq := NewScalarQuantizer(dim)
	vectors := []float32{0, 0, 0, 0, 1, 1, 1, 1}
	require.NoError(t, sq.Train(vectors, 2))

	batch := []float32{0, 0, 0, 0, 1, 1, 1, 1}
	codes, err := sq.EncodeBatch(batch, 2)
	require.NoError(t, err)
	assert.Len(t, codes, 8)

	// First vector → all 0s
	for d := 0; d < dim; d++ {
		assert.Equal(t, uint8(0), codes[d])
	}
	// Second vector → all 255s
	for d := 0; d < dim; d++ {
		assert.Equal(t, uint8(255), codes[dim+d])
	}
}

func TestSQ_ConstantDimension(t *testing.T) {
	dim := 4
	sq := NewScalarQuantizer(dim)
	// All same value in dimension 0
	vectors := []float32{5, 0, 0, 0, 5, 1, 1, 1}
	require.NoError(t, sq.Train(vectors, 2))

	code, _ := sq.Encode([]float32{5, 0.5, 0.5, 0.5})
	assert.Equal(t, uint8(0), code[0]) // constant dim → 0
}

func TestSQ_NotTrained(t *testing.T) {
	sq := NewScalarQuantizer(4)
	_, err := sq.Encode([]float32{1, 2, 3, 4})
	require.Error(t, err)
}

func TestSQ_DimensionMismatch(t *testing.T) {
	sq := NewScalarQuantizer(4)
	sq.Train([]float32{0, 0, 0, 0, 1, 1, 1, 1}, 2)
	_, err := sq.Encode([]float32{1, 2})
	require.Error(t, err)
}

func TestSQ_ReconstructionError(t *testing.T) {
	// Test that reconstruction error is bounded for random data
	dim := 128
	n := 1000
	rng := rand.New(rand.NewSource(42))

	sq := NewScalarQuantizer(dim)
	vectors := make([]float32, n*dim)
	for i := range vectors {
		vectors[i] = rng.Float32()*2 - 1
	}
	require.NoError(t, sq.Train(vectors, n))

	// Measure max per-dimension error
	maxErr := float32(0)
	for i := 0; i < n; i++ {
		vec := vectors[i*dim : (i+1)*dim]
		code, _ := sq.Encode(vec)
		decoded, _ := sq.Decode(code)
		for d := 0; d < dim; d++ {
			err := float32(math.Abs(float64(vec[d] - decoded[d])))
			if err > maxErr {
				maxErr = err
			}
		}
	}

	// Max error should be bounded by scale/2 ≈ range/(2*255) ≈ 2/(510) ≈ 0.004
	t.Logf("SQ8 max reconstruction error: %.6f", maxErr)
	assert.Less(t, maxErr, float32(0.01), "max reconstruction error should be < 0.01 for [-1,1] data")
}

// ── Benchmarks ──────────────────────────────────────────────────────────────

func BenchmarkSQ_Encode_128D(b *testing.B) {
	sq := NewScalarQuantizer(128)
	rng := rand.New(rand.NewSource(42))
	vecs := make([]float32, 1000*128)
	for i := range vecs {
		vecs[i] = rng.Float32()*2 - 1
	}
	sq.Train(vecs, 1000)

	vec := make([]float32, 128)
	for i := range vec {
		vec[i] = rng.Float32()*2 - 1
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sq.Encode(vec)
	}
}

func BenchmarkSQ_DistanceL2_128D(b *testing.B) {
	sq := NewScalarQuantizer(128)
	rng := rand.New(rand.NewSource(42))
	vecs := make([]float32, 1000*128)
	for i := range vecs {
		vecs[i] = rng.Float32()*2 - 1
	}
	sq.Train(vecs, 1000)

	query := make([]float32, 128)
	for i := range query {
		query[i] = rng.Float32()*2 - 1
	}
	code, _ := sq.Encode(query)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sq.DistanceL2SQ(query, code)
	}
}
