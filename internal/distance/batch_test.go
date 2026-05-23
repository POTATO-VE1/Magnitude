package distance

import (
	"math"
	"testing"
)

// ── Correctness Tests ───────────────────────────────────────────────────────

func TestL2BatchPure(t *testing.T) {
	// 3 vectors of dim 4
	query := []float32{1, 2, 3, 4}
	matrix := []float32{
		1, 2, 3, 4, // identical → distance 0
		2, 3, 4, 5, // diff=1 each → 1+1+1+1 = 4
		0, 0, 0, 0, // diff=1,2,3,4 → 1+4+9+16 = 30
	}
	results := make([]float32, 3)

	L2BatchPure(query, matrix, 3, 4, results)

	if results[0] != 0 {
		t.Errorf("L2(identical) = %f, want 0", results[0])
	}
	if results[1] != 4 {
		t.Errorf("L2(diff=1) = %f, want 4", results[1])
	}
	if results[2] != 30 {
		t.Errorf("L2(diff=1..4) = %f, want 30", results[2])
	}
}

func TestDotBatchPure(t *testing.T) {
	query := []float32{1, 0, 0, 0}
	matrix := []float32{
		1, 0, 0, 0, // dot = 1
		0, 1, 0, 0, // dot = 0
		1, 1, 1, 1, // dot = 1
	}
	results := make([]float32, 3)

	DotBatchPure(query, matrix, 3, 4, results)

	if results[0] != 1 {
		t.Errorf("Dot(unit) = %f, want 1", results[0])
	}
	if results[1] != 0 {
		t.Errorf("Dot(orthogonal) = %f, want 0", results[1])
	}
	if results[2] != 1 {
		t.Errorf("Dot(mixed) = %f, want 1", results[2])
	}
}

func TestCosineBatchPure(t *testing.T) {
	query := []float32{1, 0, 0}
	matrix := []float32{
		1, 0, 0, // identical → distance 0
		0, 1, 0, // orthogonal → distance 1
		-1, 0, 0, // opposite → distance 2
	}
	results := make([]float32, 3)

	CosineBatchPure(query, matrix, 3, 3, results)

	if math.Abs(float64(results[0])) > 1e-6 {
		t.Errorf("Cosine(identical) = %f, want ~0", results[0])
	}
	if math.Abs(float64(results[1])-1.0) > 1e-6 {
		t.Errorf("Cosine(orthogonal) = %f, want ~1", results[1])
	}
	if math.Abs(float64(results[2])-2.0) > 1e-6 {
		t.Errorf("Cosine(opposite) = %f, want ~2", results[2])
	}
}

func TestManhattanBatchPure(t *testing.T) {
	query := []float32{1, 2, 3, 4}
	matrix := []float32{
		1, 2, 3, 4, // identical → 0
		2, 3, 4, 5, // |1|+|1|+|1|+|1| = 4
		0, 0, 0, 0, // |1|+|2|+|3|+|4| = 10
	}
	results := make([]float32, 3)

	ManhattanBatchPure(query, matrix, 3, 4, results)

	if results[0] != 0 {
		t.Errorf("Manhattan(identical) = %f, want 0", results[0])
	}
	if results[1] != 4 {
		t.Errorf("Manhattan(diff=1) = %f, want 4", results[1])
	}
	if results[2] != 10 {
		t.Errorf("Manhattan(diff=1..4) = %f, want 10", results[2])
	}
}

func TestBatchDistance(t *testing.T) {
	query := []float32{1, 2, 3}
	matrix := []float32{1, 2, 3, 4, 5, 6}
	results := make([]float32, 2)

	BatchDistance(query, matrix, 2, 3, "l2", results)
	if results[0] != 0 {
		t.Errorf("BatchDistance L2(identical) = %f, want 0", results[0])
	}

	BatchDistance(query, matrix, 2, 3, "dot", results)
	if results[0] != 14 { // 1+4+9
		t.Errorf("BatchDistance Dot(self) = %f, want 14", results[0])
	}

	BatchDistance(query, matrix, 2, 3, "cosine", results)
	if math.Abs(float64(results[0])) > 1e-6 {
		t.Errorf("BatchDistance Cosine(self) = %f, want ~0", results[0])
	}

	BatchDistance(query, matrix, 2, 3, "manhattan", results)
	if results[0] != 0 {
		t.Errorf("BatchDistance Manhattan(self) = %f, want 0", results[0])
	}
}

// ── Edge Cases ──────────────────────────────────────────────────────────────

func TestBatchEmptyMatrix(t *testing.T) {
	query := []float32{1, 2, 3}
	matrix := []float32{}
	results := make([]float32, 0)

	L2BatchPure(query, matrix, 0, 3, results)
	// Should not panic
}

func TestBatchDimensionMismatch(t *testing.T) {
	// Dim 4 but vectors are dim 3 — caller's responsibility, but shouldn't panic
	query := []float32{1, 2, 3, 4}
	matrix := []float32{1, 2, 3, 4, 5, 6}
	results := make([]float32, 2)

	// This will read past matrix bounds if dim check isn't there
	// We test that it at least doesn't panic with correct n/dim
	L2BatchPure(query, matrix, 2, 3, results)
}

func TestBatchSingleVector(t *testing.T) {
	query := []float32{1, 2, 3}
	matrix := []float32{4, 5, 6}
	results := make([]float32, 1)

	L2BatchPure(query, matrix, 1, 3, results)
	expected := float32(9 + 9 + 9) // (4-1)^2 + (5-2)^2 + (6-3)^2
	if results[0] != expected {
		t.Errorf("L2 single = %f, want %f", results[0], expected)
	}
}

func TestBatchLargeDimension(t *testing.T) {
	// 768-dim vectors (SigLIP dimension)
	dim := 768
	n := 10
	query := make([]float32, dim)
	matrix := make([]float32, n*dim)
	for i := range query {
		query[i] = float32(i)
	}
	for i := range matrix {
		matrix[i] = float32(i % dim)
	}
	results := make([]float32, n)

	L2BatchPure(query, matrix, n, dim, results)

	// First vector is identical to query
	if results[0] != 0 {
		t.Errorf("L2 768-dim identical = %f, want 0", results[0])
	}
	// All results should be non-negative
	for i, r := range results {
		if r < 0 {
			t.Errorf("L2 768-dim [%d] = %f, want >= 0", i, r)
		}
	}
}

func TestL2BatchDispatchMatchesPure(t *testing.T) {
	// Simplest case: dim=8, n=1
	query := []float32{0, 1, 2, 3, 4, 5, 6, 7}
	matrix := []float32{8, 9, 10, 11, 12, 13, 14, 15}

	pure := make([]float32, 1)
	disp := make([]float32, 1)
	L2BatchPure(query, matrix, 1, 8, pure)
	L2Batch(query, matrix, 1, 8, disp)

	t.Logf("dim=8 n=1: pure=%f dispatch=%f", pure[0], disp[0])
	if pure[0] != disp[0] {
		t.Errorf("dim=8 n=1 mismatch: pure=%f dispatch=%f", pure[0], disp[0])
	}

	// dim=16, n=1
	query2 := make([]float32, 16)
	matrix2 := make([]float32, 16)
	for i := range query2 {
		query2[i] = float32(i)
		matrix2[i] = float32(i + 10)
	}
	L2BatchPure(query2, matrix2, 1, 16, pure)
	L2Batch(query2, matrix2, 1, 16, disp)
	t.Logf("dim=16 n=1: pure=%f dispatch=%f", pure[0], disp[0])
	if pure[0] != disp[0] {
		t.Errorf("dim=16 n=1 mismatch: pure=%f dispatch=%f", pure[0], disp[0])
	}

	// dim=8, n=3 (multiple vectors)
	pure3 := make([]float32, 3)
	disp3 := make([]float32, 3)
	matrix3 := []float32{
		8, 9, 10, 11, 12, 13, 14, 15, // vector 0
		0, 1, 2, 3, 4, 5, 6, 7, // vector 1 (identical to query)
		1, 2, 3, 4, 5, 6, 7, 8, // vector 2
	}
	L2BatchPure(query, matrix3, 3, 8, pure3)
	L2Batch(query, matrix3, 3, 8, disp3)
	t.Logf("dim=8 n=3: pure=%v dispatch=%v", pure3, disp3)
	for i := 0; i < 3; i++ {
		if pure3[i] != disp3[i] {
			t.Errorf("dim=8 n=3 [%d]: pure=%f dispatch=%f", i, pure3[i], disp3[i])
		}
	}

	// dim=128, n=3 (large dim, multiple of 8)
	dim128 := 128
	q128 := make([]float32, dim128)
	m128 := make([]float32, 3*dim128)
	for i := range q128 {
		q128[i] = float32(i)
	}
	for i := range m128 {
		m128[i] = float32(i + 10)
	}
	p128 := make([]float32, 3)
	d128 := make([]float32, 3)
	L2BatchPure(q128, m128, 3, dim128, p128)
	L2Batch(q128, m128, 3, dim128, d128)
	t.Logf("dim=128 n=3: pure=%v dispatch=%v", p128, d128)
	for i := 0; i < 3; i++ {
		if p128[i] != d128[i] {
			t.Errorf("dim=128 n=3 [%d]: pure=%f dispatch=%f", i, p128[i], d128[i])
		}
	}

	// dim=5 (NOT multiple of 8 — tests scalar tail)
	query5 := []float32{1, 2, 3, 4, 5}
	matrix5 := []float32{
		6, 7, 8, 9, 10,
		1, 2, 3, 4, 5, // identical
	}
	p5 := make([]float32, 2)
	d5 := make([]float32, 2)
	L2BatchPure(query5, matrix5, 2, 5, p5)
	L2Batch(query5, matrix5, 2, 5, d5)
	t.Logf("dim=5 n=2: pure=%v dispatch=%v", p5, d5)
	for i := 0; i < 2; i++ {
		if p5[i] != d5[i] {
			t.Errorf("dim=5 n=2 [%d]: pure=%f dispatch=%f", i, p5[i], d5[i])
		}
	}

	// n=2, dim=8 (exactly 1 AVX2 iteration per vector)
	query8 := []float32{10, 20, 30, 40, 50, 60, 70, 80}
	matrix8 := []float32{
		10, 20, 30, 40, 50, 60, 70, 80, // identical
		11, 22, 33, 44, 55, 66, 77, 88, // different
	}
	p8 := make([]float32, 2)
	d8 := make([]float32, 2)
	L2BatchPure(query8, matrix8, 2, 8, p8)
	L2Batch(query8, matrix8, 2, 8, d8)
	t.Logf("dim=8 n=2: pure=%v dispatch=%v", p8, d8)
	for i := 0; i < 2; i++ {
		if p8[i] != d8[i] {
			t.Errorf("dim=8 n=2 [%d]: pure=%f dispatch=%f", i, p8[i], d8[i])
		}
	}

	// Larger test: dim=128, n=100, 0.1 scaling
	dimBig := 128
	nBig := 100
	qBig := make([]float32, dimBig)
	mBig := make([]float32, nBig*dimBig)
	for i := range qBig {
		qBig[i] = float32(i) * 0.1
	}
	for i := range mBig {
		mBig[i] = float32(i) * 0.01
	}
	pBig := make([]float32, nBig)
	dBig := make([]float32, nBig)
	L2BatchPure(qBig, mBig, nBig, dimBig, pBig)
	L2Batch(qBig, mBig, nBig, dimBig, dBig)

	// Print first 10 and check pattern
	for i := 0; i < 10; i++ {
		t.Logf("[%d] pure=%.3f dispatch=%.3f ratio=%.4f", i, pBig[i], dBig[i], dBig[i]/pBig[i])
	}
}

func TestDotBatchDispatchMatchesPure(t *testing.T) {
	// Simplest case: dim=8, n=1
	query := []float32{0, 1, 2, 3, 4, 5, 6, 7}
	matrix := []float32{8, 9, 10, 11, 12, 13, 14, 15}

	pure := make([]float32, 1)
	disp := make([]float32, 1)
	DotBatchPure(query, matrix, 1, 8, pure)
	DotBatch(query, matrix, 1, 8, disp)

	t.Logf("dot dim=8 n=1: pure=%f dispatch=%f", pure[0], disp[0])
	if pure[0] != disp[0] {
		t.Errorf("dot dim=8 n=1 mismatch: pure=%f dispatch=%f", pure[0], disp[0])
	}
}

// ── Benchmarks ──────────────────────────────────────────────────────────────

func BenchmarkL2BatchPure(b *testing.B) {
	dim := 768
	n := 1000
	query := make([]float32, dim)
	matrix := make([]float32, n*dim)
	results := make([]float32, n)
	for i := range query {
		query[i] = float32(i)
	}
	for i := range matrix {
		matrix[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		L2BatchPure(query, matrix, n, dim, results)
	}
}

func BenchmarkL2BatchDispatch(b *testing.B) {
	dim := 768
	n := 1000
	query := make([]float32, dim)
	matrix := make([]float32, n*dim)
	results := make([]float32, n)
	for i := range query {
		query[i] = float32(i)
	}
	for i := range matrix {
		matrix[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		L2Batch(query, matrix, n, dim, results)
	}
}

func BenchmarkDotBatchDispatch(b *testing.B) {
	dim := 768
	n := 1000
	query := make([]float32, dim)
	matrix := make([]float32, n*dim)
	results := make([]float32, n)
	for i := range query {
		query[i] = float32(i)
	}
	for i := range matrix {
		matrix[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DotBatch(query, matrix, n, dim, results)
	}
}

func BenchmarkDotBatchPure(b *testing.B) {
	dim := 768
	n := 1000
	query := make([]float32, dim)
	matrix := make([]float32, n*dim)
	results := make([]float32, n)
	for i := range query {
		query[i] = float32(i)
	}
	for i := range matrix {
		matrix[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DotBatchPure(query, matrix, n, dim, results)
	}
}

func BenchmarkCosineBatchPure(b *testing.B) {
	dim := 768
	n := 1000
	query := make([]float32, dim)
	matrix := make([]float32, n*dim)
	results := make([]float32, n)
	for i := range query {
		query[i] = float32(i)
	}
	for i := range matrix {
		matrix[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CosineBatchPure(query, matrix, n, dim, results)
	}
}

func BenchmarkManhattanBatchPure(b *testing.B) {
	dim := 768
	n := 1000
	query := make([]float32, dim)
	matrix := make([]float32, n*dim)
	results := make([]float32, n)
	for i := range query {
		query[i] = float32(i)
	}
	for i := range matrix {
		matrix[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ManhattanBatchPure(query, matrix, n, dim, results)
	}
}

// Benchmark per-vector L2Squared (old path) for comparison
func BenchmarkL2SquaredPerVector(b *testing.B) {
	dim := 768
	n := 1000
	query := make([]float32, dim)
	matrix := make([]float32, n*dim)
	for i := range query {
		query[i] = float32(i)
	}
	for i := range matrix {
		matrix[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < n; j++ {
			row := matrix[j*dim : (j+1)*dim]
			L2Squared(query, row)
		}
	}
}
