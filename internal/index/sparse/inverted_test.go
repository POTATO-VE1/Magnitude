package sparse

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ── Tokenizer Tests ─────────────────────────────────────────────────────────

func TestTokenize(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  []string
	}{
		{"simple", "hello world", []string{"hello", "world"}},
		{"removes stop words", "the quick brown fox", []string{"quick", "brown", "fox"}},
		{"lowercase", "SQL Query Optimization", []string{"sql", "query", "optimization"}},
		{"punctuation", "hello, world! how's it going?", []string{"hello", "world", "how", "going"}},
		{"single chars stripped", "a b c hello", []string{"hello"}},
		{"empty", "", nil},
		{"unicode", "café résumé", []string{"café", "résumé"}},
		{"numbers", "vector128 dim768", []string{"vector128", "dim768"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Tokenize(tt.input)
			if tt.want == nil {
				assert.Empty(t, got)
			} else {
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

// ── InvertedIndex Tests ─────────────────────────────────────────────────────

func TestNewInvertedIndex(t *testing.T) {
	idx := NewInvertedIndex()
	assert.Equal(t, 0, idx.Len())
	assert.Equal(t, 0, idx.VocabSize())
}

func TestAddDocument(t *testing.T) {
	idx := NewInvertedIndex()
	idx.AddDocument(1, "the quick brown fox jumps over the lazy dog")
	idx.AddDocument(2, "the quick brown cat sits on the mat")

	assert.Equal(t, 2, idx.Len())
	assert.Greater(t, idx.VocabSize(), 0)
}

func TestSearch_SingleTerm(t *testing.T) {
	idx := NewInvertedIndex()
	idx.AddDocument(1, "machine learning neural networks deep learning")
	idx.AddDocument(2, "database indexing query optimization")
	idx.AddDocument(3, "machine learning algorithms random forest")

	results := idx.Search("machine learning", 10)
	require.Greater(t, len(results), 0)

	// Docs 1 and 3 contain "machine learning"; doc 2 does not
	foundIDs := make(map[uint64]bool)
	for _, r := range results {
		foundIDs[r.DocID] = true
		assert.Greater(t, r.Score, float32(0))
	}
	assert.True(t, foundIDs[1], "doc 1 should match 'machine learning'")
	assert.True(t, foundIDs[3], "doc 3 should match 'machine learning'")
	assert.False(t, foundIDs[2], "doc 2 should not match 'machine learning'")
}

func TestSearch_TopK(t *testing.T) {
	idx := NewInvertedIndex()
	for i := uint64(0); i < 100; i++ {
		idx.AddDocument(i, "vector database search algorithm optimization")
	}

	results := idx.Search("vector database", 5)
	assert.Len(t, results, 5)
}

func TestSearch_NoMatch(t *testing.T) {
	idx := NewInvertedIndex()
	idx.AddDocument(1, "hello world")

	results := idx.Search("quantum physics", 10)
	assert.Empty(t, results)
}

func TestSearch_EmptyIndex(t *testing.T) {
	idx := NewInvertedIndex()
	results := idx.Search("anything", 10)
	assert.Empty(t, results)
}

func TestSearch_EmptyQuery(t *testing.T) {
	idx := NewInvertedIndex()
	idx.AddDocument(1, "hello world")
	results := idx.Search("", 10)
	assert.Empty(t, results)
}

func TestSearch_BM25Ranking(t *testing.T) {
	idx := NewInvertedIndex()
	// Doc 1: high TF for "database"
	idx.AddDocument(1, "database database database indexing")
	// Doc 2: low TF for "database"
	idx.AddDocument(2, "database indexing search optimization vector storage")

	results := idx.Search("database", 10)
	require.Len(t, results, 2)
	// Doc 1 should rank higher (higher TF for "database")
	assert.Equal(t, uint64(1), results[0].DocID)
	assert.Greater(t, results[0].Score, results[1].Score)
}

func TestSearch_IDFEffect(t *testing.T) {
	idx := NewInvertedIndex()
	// "common" appears in all docs, "rare" appears in one
	idx.AddDocument(1, "common word rare word")
	idx.AddDocument(2, "common word another word")
	idx.AddDocument(3, "common word yet another word")

	// Searching for "rare" should return only doc 1 with a high score
	results := idx.Search("rare", 10)
	require.Len(t, results, 1)
	assert.Equal(t, uint64(1), results[0].DocID)

	// "rare" has higher IDF than "common", so searching for both
	// should still rank doc 1 first
	results = idx.Search("common rare", 10)
	require.Greater(t, len(results), 0)
	assert.Equal(t, uint64(1), results[0].DocID)
}

func TestRemoveDocument(t *testing.T) {
	idx := NewInvertedIndex()
	idx.AddDocument(1, "hello world")
	idx.AddDocument(2, "hello universe")

	idx.RemoveDocument(1)
	assert.Equal(t, 1, idx.Len())

	// Search should only find doc 2
	results := idx.Search("hello", 10)
	require.Len(t, results, 1)
	assert.Equal(t, uint64(2), results[0].DocID)
}

func TestRemoveDocument_NonExistent(t *testing.T) {
	idx := NewInvertedIndex()
	idx.AddDocument(1, "hello")
	idx.RemoveDocument(999) // should not panic
	assert.Equal(t, 1, idx.Len())
}

func TestCustomParams(t *testing.T) {
	idx := NewInvertedIndexWithParams(1.5, 0.5)
	idx.AddDocument(1, "hello world")
	results := idx.Search("hello", 10)
	require.Len(t, results, 1)
}

// ── Benchmark ───────────────────────────────────────────────────────────────

func BenchmarkBM25_Search_10K(b *testing.B) {
	idx := NewInvertedIndex()
	docs := []string{
		"machine learning deep neural networks",
		"database query optimization indexing",
		"vector similarity search algorithms",
		"natural language processing transformers",
		"distributed systems consensus protocol",
	}
	for i := uint64(0); i < 10000; i++ {
		idx.AddDocument(i, docs[i%5])
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		idx.Search("machine learning", 10)
	}
}
