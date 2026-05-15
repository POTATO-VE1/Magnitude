// Package index defines the core Index interface that every index type
// (Flat, IVF, HNSW, SPANN) must implement. API handlers never know which
// implementation is underneath — they only speak this interface.
//
// This file is the backbone of the entire system and NEVER changes after Phase 0.
// Every architectural decision about adding capabilities must flow through
// this interface definition.
package index

import "context"

// Index is the single most important contract in the codebase.
// EVERY index type (Flat, IVF, HNSW, SPANN) implements it.
// API handlers NEVER know which implementation is underneath.
type Index interface {
	// Insert adds a vector with the given ID into the index.
	// Returns ErrDuplicateID if the ID already exists.
	// Returns ErrDimensionMismatch if len(vector) != index dimension.
	Insert(id uint64, vector []float32) error

	// Search performs an approximate nearest-neighbor search for the query vector,
	// returning up to k results. nprobe controls the search quality/speed tradeoff
	// (relevant for IVF and SPANN; ignored by Flat which always searches everything).
	Search(ctx context.Context, query []float32, k int, nprobe int) ([]SearchResult, error)

	// Delete marks a vector as deleted. The vector may remain in memory
	// until the next Rebuild or compaction cycle.
	// Returns ErrVectorNotFound if the ID does not exist.
	Delete(id uint64) error

	// Len returns the number of live (non-deleted) vectors in the index.
	Len() int

	// Rebuild triggers a full index reconstruction. For IVF this re-runs K-Means;
	// for HNSW this re-inserts all vectors. For Flat this is a no-op.
	// Called automatically when the dirty fraction exceeds DirtyThreshold.
	Rebuild() error

	// Flush persists any in-memory state to disk. Must be called before
	// process exit to avoid data loss.
	Flush() error
}

// SearchResult represents a single result from a nearest-neighbor search.
type SearchResult struct {
	// ID is the unique identifier of the matched vector.
	ID uint64

	// Distance is the raw distance metric value (L2, cosine distance, etc.).
	// Lower is better for L2 and cosine distance.
	Distance float32

	// Score is a normalized similarity score in [0, 1].
	// For cosine: Score = 1 - normalized_distance (higher = more similar).
	// For L2: Score = 1 / (1 + Distance) (higher = more similar).
	Score float32
}

// VectorExporter is an optional interface that index implementations can
// satisfy to support vector enumeration for migration. This does NOT
// modify the frozen Index interface — it is checked via type assertion:
//
//	if exporter, ok := idx.(index.VectorExporter); ok { ... }
type VectorExporter interface {
	// ExportVectors returns all live (non-deleted) vectors in the index.
	// Used by the migration engine to stream vectors to a target node.
	ExportVectors() []ExportedVector
}

// ExportedVector is a vector with its ID, suitable for migration transfer.
type ExportedVector struct {
	ID     uint64
	Vector []float32
}

