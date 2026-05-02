package quantize

import (
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPQ_New(t *testing.T) {
	pq, err := NewProductQuantizer(128, 8)
	require.NoError(t, err)
	assert.Equal(t, 128, pq.Dim)
	assert.Equal(t, 8, pq.M)
	assert.Equal(t, 16, pq.SubDim)
	assert.Equal(t, 256, pq.KBook)
}

func TestPQ_New_Invalid(t *testing.T) {
	_, err := NewProductQuantizer(0, 8)
	require.Error(t, err)

	_, err = NewProductQuantizer(128, 0)
	require.Error(t, err)

	// Dim not divisible by M
	_, err = NewProductQuantizer(127, 8)
	require.Error(t, err)
}

func TestPQ_TrainAndEncode(t *testing.T) {
	dim := 8
	m := 4
	pq, _ := NewProductQuantizer(dim, m)

	// Generate training data
	rng := rand.New(rand.NewSource(42))
	n := 500
	vectors := make([]float32, n*dim)
	for i := range vectors {
		vectors[i] = rng.Float32()*2 - 1
	}

	require.NoError(t, pq.Train(vectors, n, 10))
	assert.True(t, pq.IsTrained())

	// Encode a vector
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rng.Float32()*2 - 1
	}
	codes, err := pq.Encode(vec)
	require.NoError(t, err)
	assert.Len(t, codes, m)
}

func TestPQ_Decode(t *testing.T) {
	dim := 8
	m := 4
	pq, _ := NewProductQuantizer(dim, m)

	rng := rand.New(rand.NewSource(42))
	n := 500
	vectors := make([]float32, n*dim)
	for i := range vectors {
		vectors[i] = rng.Float32()*2 - 1
	}
	require.NoError(t, pq.Train(vectors, n, 10))

	// Encode then decode
	original := vectors[:dim]
	codes, _ := pq.Encode(original)
	decoded, err := pq.Decode(codes)
	require.NoError(t, err)
	assert.Len(t, decoded, dim)

	// Decoded should be an approximation (not exact)
	var totalErr float32
	for i := range original {
		diff := original[i] - decoded[i]
		totalErr += diff * diff
	}
	rmse := float32(math.Sqrt(float64(totalErr / float32(dim))))
	t.Logf("PQ decode RMSE = %.6f (M=%d, dim=%d)", rmse, m, dim)
	assert.Less(t, rmse, float32(1.0), "PQ decode RMSE should be reasonable")
}

func TestPQ_ADC(t *testing.T) {
	dim := 16
	m := 4
	pq, _ := NewProductQuantizer(dim, m)

	rng := rand.New(rand.NewSource(42))
	n := 500
	vectors := make([]float32, n*dim)
	for i := range vectors {
		vectors[i] = rng.Float32()*2 - 1
	}
	require.NoError(t, pq.Train(vectors, n, 10))

	// Build distance table for a query
	query := make([]float32, dim)
	for i := range query {
		query[i] = rng.Float32()*2 - 1
	}
	table := pq.BuildDistanceTable(query)
	assert.Len(t, table, m*256)

	// Compute ADC distance to an encoded vector
	vec := vectors[:dim]
	codes, _ := pq.Encode(vec)
	adcDist := pq.DistanceADC(table, codes)

	// Compare with exact L2 distance
	var exactDist float32
	for i := 0; i < dim; i++ {
		diff := query[i] - vec[i]
		exactDist += diff * diff
	}

	t.Logf("ADC distance: %.4f, exact L2²: %.4f", adcDist, exactDist)
	// ADC should be a reasonable approximation
	assert.InDelta(t, exactDist, adcDist, float64(exactDist)*0.5+1.0,
		"ADC distance should be a reasonable approximation of exact L2²")
}

func TestPQ_CompressionRatio(t *testing.T) {
	pq, _ := NewProductQuantizer(128, 8)
	assert.Equal(t, 64.0, pq.CompressionRatio()) // 4*128/8 = 64x
}

func TestPQ_NotTrained(t *testing.T) {
	pq, _ := NewProductQuantizer(16, 4)
	_, err := pq.Encode(make([]float32, 16))
	require.Error(t, err)
}

func TestPQ_DimensionMismatch(t *testing.T) {
	pq, _ := NewProductQuantizer(16, 4)
	rng := rand.New(rand.NewSource(42))
	vecs := make([]float32, 100*16)
	for i := range vecs {
		vecs[i] = rng.Float32()
	}
	pq.Train(vecs, 100, 5)

	_, err := pq.Encode(make([]float32, 8))
	require.Error(t, err)
}

func TestPQ_ADC_RankingCorrelation(t *testing.T) {
	// Verify that ADC rankings correlate with exact L2 rankings
	dim := 32
	m := 8
	pq, _ := NewProductQuantizer(dim, m)

	rng := rand.New(rand.NewSource(42))
	n := 1000
	vectors := make([]float32, n*dim)
	for i := range vectors {
		vectors[i] = rng.Float32()*2 - 1
	}
	require.NoError(t, pq.Train(vectors, n, 15))

	// Encode all vectors
	codes := make([][]uint8, n)
	for i := 0; i < n; i++ {
		codes[i], _ = pq.Encode(vectors[i*dim : (i+1)*dim])
	}

	// Query
	query := make([]float32, dim)
	for i := range query {
		query[i] = rng.Float32()*2 - 1
	}
	table := pq.BuildDistanceTable(query)

	// Find top-10 by both methods
	type distPair struct {
		idx  int
		dist float32
	}

	// Exact distances
	exactDists := make([]distPair, n)
	for i := 0; i < n; i++ {
		var d float32
		for j := 0; j < dim; j++ {
			diff := query[j] - vectors[i*dim+j]
			d += diff * diff
		}
		exactDists[i] = distPair{i, d}
	}

	// ADC distances
	adcDists := make([]distPair, n)
	for i := 0; i < n; i++ {
		adcDists[i] = distPair{i, pq.DistanceADC(table, codes[i])}
	}

	// Sort both (selection sort for top-10)
	for k := 0; k < 10; k++ {
		for j := k + 1; j < n; j++ {
			if exactDists[j].dist < exactDists[k].dist {
				exactDists[j], exactDists[k] = exactDists[k], exactDists[j]
			}
			if adcDists[j].dist < adcDists[k].dist {
				adcDists[j], adcDists[k] = adcDists[k], adcDists[j]
			}
		}
	}

	// Count overlap in top-10
	exactTop10 := make(map[int]bool)
	for k := 0; k < 10; k++ {
		exactTop10[exactDists[k].idx] = true
	}
	hits := 0
	for k := 0; k < 10; k++ {
		if exactTop10[adcDists[k].idx] {
			hits++
		}
	}
	recall := float64(hits) / 10.0
	t.Logf("PQ ADC Recall@10 = %.2f (M=%d, dim=%d, n=%d)", recall, m, dim, n)
	assert.GreaterOrEqual(t, recall, 0.30, "PQ ADC recall should show some ranking correlation")
}

// ── Benchmarks ──────────────────────────────────────────────────────────────

func BenchmarkPQ_Encode_128D_M8(b *testing.B) {
	pq, _ := NewProductQuantizer(128, 8)
	rng := rand.New(rand.NewSource(42))
	n := 1000
	vecs := make([]float32, n*128)
	for i := range vecs {
		vecs[i] = rng.Float32()*2 - 1
	}
	pq.Train(vecs, n, 10)

	vec := make([]float32, 128)
	for i := range vec {
		vec[i] = rng.Float32()*2 - 1
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pq.Encode(vec)
	}
}

func BenchmarkPQ_ADC_128D_M8(b *testing.B) {
	pq, _ := NewProductQuantizer(128, 8)
	rng := rand.New(rand.NewSource(42))
	n := 1000
	vecs := make([]float32, n*128)
	for i := range vecs {
		vecs[i] = rng.Float32()*2 - 1
	}
	pq.Train(vecs, n, 10)

	query := make([]float32, 128)
	for i := range query {
		query[i] = rng.Float32()*2 - 1
	}
	table := pq.BuildDistanceTable(query)
	codes, _ := pq.Encode(vecs[:128])

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pq.DistanceADC(table, codes)
	}
}
