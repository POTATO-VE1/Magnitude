// Package sparse implements a BM25-scored inverted index for keyword search.
//
// This is the "sparse" half of hybrid search (dense + sparse + RRF fusion).
// Dense search (HNSW/SPANN) finds semantically similar vectors.
// Sparse search (BM25) finds documents containing exact query terms.
// Combined via Reciprocal Rank Fusion, they outperform either alone.
//
// Architecture:
//
//	Document → Tokenize → Term IDs → Inverted Index → Posting Lists
//	                                                      ↓
//	Query    → Tokenize → Term IDs → BM25 Score per doc → Top-K
//
// BM25 formula (Robertson variant):
//
//	score(q, d) = Σ IDF(qi) · (tf(qi, d) · (k1 + 1)) / (tf(qi, d) + k1 · (1 - b + b · |d|/avgdl))
//	IDF(qi) = log(1 + (N - df(qi) + 0.5) / (df(qi) + 0.5))
package sparse

import (
	"math"
	"sort"
	"strings"
	"sync"
	"unicode"
)

// Posting represents one document's term occurrence in the inverted index.
type Posting struct {
	DocID  uint64
	RawTF  float32 // raw term frequency in this document
}

// PostingList is a sorted list of postings for a single term.
type PostingList []Posting

// SearchResult is a scored document from BM25 search.
type SearchResult struct {
	DocID uint64
	Score float32
}

// InvertedIndex implements a BM25-scored inverted index for keyword search.
type InvertedIndex struct {
	mu           sync.RWMutex
	postings     map[uint32]PostingList // term_id → postings
	docLengths   map[uint64]int         // docID → number of terms
	termDF       map[uint32]int         // term_id → document frequency
	totalDocs    int                    // N in BM25 formula
	avgDocLen    float64                // avgdl in BM25 formula
	vocab        map[string]uint32      // term string → term_id
	nextTermID   uint32

	// BM25 parameters (industry standard defaults)
	k1 float64 // term saturation; default: 1.2
	b  float64 // length normalization; default: 0.75
}

// NewInvertedIndex creates a new BM25 inverted index with default parameters.
func NewInvertedIndex() *InvertedIndex {
	return &InvertedIndex{
		postings:   make(map[uint32]PostingList),
		docLengths: make(map[uint64]int),
		termDF:     make(map[uint32]int),
		vocab:      make(map[string]uint32),
		k1:         1.2,
		b:          0.75,
	}
}

// NewInvertedIndexWithParams creates an inverted index with custom BM25 parameters.
func NewInvertedIndexWithParams(k1, b float64) *InvertedIndex {
	idx := NewInvertedIndex()
	idx.k1 = k1
	idx.b = b
	return idx
}

// ── Tokenizer ───────────────────────────────────────────────────────────────

// Tokenize converts text into a slice of normalized term strings.
// Applies: lowercase, unicode normalization, punctuation stripping, stop-word removal.
func Tokenize(text string) []string {
	text = strings.ToLower(text)
	words := strings.FieldsFunc(text, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})

	result := make([]string, 0, len(words))
	for _, w := range words {
		if len(w) < 2 {
			continue // skip single-char tokens
		}
		if stopWords[w] {
			continue
		}
		result = append(result, w)
	}
	return result
}

// stopWords is a minimal English stop-word set.
var stopWords = map[string]bool{
	"the": true, "is": true, "at": true, "which": true, "on": true,
	"a": true, "an": true, "and": true, "or": true, "but": true,
	"in": true, "with": true, "to": true, "for": true, "of": true,
	"it": true, "by": true, "from": true, "as": true, "be": true,
	"was": true, "are": true, "were": true, "been": true, "has": true,
	"have": true, "had": true, "do": true, "does": true, "did": true,
	"will": true, "would": true, "could": true, "should": true,
	"this": true, "that": true, "these": true, "those": true,
	"not": true, "no": true, "so": true, "if": true, "then": true,
}

// ── Index Operations ────────────────────────────────────────────────────────

// termID returns the ID for a term, creating a new one if needed.
func (idx *InvertedIndex) termID(term string) uint32 {
	id, ok := idx.vocab[term]
	if !ok {
		id = idx.nextTermID
		idx.vocab[term] = id
		idx.nextTermID++
	}
	return id
}

// AddDocument indexes a document for BM25 search.
func (idx *InvertedIndex) AddDocument(docID uint64, text string) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	terms := Tokenize(text)
	idx.docLengths[docID] = len(terms)
	idx.totalDocs++

	// Count term frequencies
	tf := make(map[uint32]int)
	for _, t := range terms {
		tid := idx.termID(t)
		tf[tid]++
	}

	// Add to posting lists
	for termID, freq := range tf {
		idx.postings[termID] = append(idx.postings[termID], Posting{
			DocID: docID,
			RawTF: float32(freq),
		})
		idx.termDF[termID]++
	}

	// Update average document length (online mean)
	n := float64(idx.totalDocs)
	idx.avgDocLen = idx.avgDocLen*(n-1)/n + float64(len(terms))/n
}

// RemoveDocument removes a document from the index.
func (idx *InvertedIndex) RemoveDocument(docID uint64) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if _, exists := idx.docLengths[docID]; !exists {
		return
	}

	// Remove from all posting lists
	for termID, postings := range idx.postings {
		for i, p := range postings {
			if p.DocID == docID {
				idx.postings[termID] = append(postings[:i], postings[i+1:]...)
				idx.termDF[termID]--
				if idx.termDF[termID] == 0 {
					delete(idx.termDF, termID)
					delete(idx.postings, termID)
				}
				break
			}
		}
	}

	// Update stats
	oldLen := idx.docLengths[docID]
	delete(idx.docLengths, docID)
	idx.totalDocs--
	if idx.totalDocs > 0 {
		n := float64(idx.totalDocs)
		idx.avgDocLen = (idx.avgDocLen*(n+1) - float64(oldLen)) / n
	} else {
		idx.avgDocLen = 0
	}
}

// ── BM25 Scoring ────────────────────────────────────────────────────────────

// bm25Score computes the BM25 score for a single query term against a document.
func (idx *InvertedIndex) bm25Score(termID uint32, rawTF float32, docLen int) float32 {
	df := float64(idx.termDF[termID])
	N := float64(idx.totalDocs)

	// Robertson IDF: log(1 + (N - df + 0.5) / (df + 0.5))
	idf := math.Log(1 + (N-df+0.5)/(df+0.5))

	// BM25 TF normalization
	tf := float64(rawTF)
	dl := float64(docLen)
	tfNorm := tf * (idx.k1 + 1) / (tf + idx.k1*(1-idx.b+idx.b*dl/idx.avgDocLen))

	return float32(idf * tfNorm)
}

// Search returns the top-K documents matching the query text, ranked by BM25 score.
func (idx *InvertedIndex) Search(query string, k int) []SearchResult {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.totalDocs == 0 || k <= 0 {
		return nil
	}

	queryTerms := Tokenize(query)
	if len(queryTerms) == 0 {
		return nil
	}

	// Accumulate BM25 scores across all query terms
	scores := make(map[uint64]float32)
	for _, term := range queryTerms {
		termID, ok := idx.vocab[term]
		if !ok {
			continue // query term not in corpus
		}
		postings, ok := idx.postings[termID]
		if !ok {
			continue
		}
		for _, p := range postings {
			docLen := idx.docLengths[p.DocID]
			scores[p.DocID] += idx.bm25Score(termID, p.RawTF, docLen)
		}
	}

	// Collect and sort by score descending
	results := make([]SearchResult, 0, len(scores))
	for docID, score := range scores {
		results = append(results, SearchResult{DocID: docID, Score: score})
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if len(results) > k {
		results = results[:k]
	}
	return results
}

// Len returns the number of indexed documents.
func (idx *InvertedIndex) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.totalDocs
}

// VocabSize returns the number of unique terms.
func (idx *InvertedIndex) VocabSize() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.vocab)
}
