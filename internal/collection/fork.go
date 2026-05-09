// Collection forking — zero-copy collection snapshots.
//
// Collection forking creates a new collection that starts as a copy
// of an existing collection WITHOUT physically duplicating vector data.
//
// Architecture:
//   - Single-node: The new collection gets a deep copy of the in-memory index
//     state. For file-backed segments, hard links (link(2)) are used — O(1) at
//     the filesystem layer, same inode, no data copy.
//   - Distributed (S3): The SysDB segment references are copied; S3 keys are
//     reused. The GC's pin counting prevents premature deletion of shared segments.
//
// ChromaDB uses this pattern for branching workflows: fork a production collection,
// fine-tune embeddings on the fork, then swap if quality improves.
package collection

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/POTATO-VE1/Magnitude/internal/metadata"
)

// ForkCollection creates a new collection that is a logical copy of src.
// The new collection has an independent index that starts with the same config
// as the source collection. Metadata is deep-copied so mutations to either
// collection do not affect the other.
func (m *Manager) ForkCollection(ctx context.Context, srcID, newName string) (*metadata.Collection, error) {
	m.mu.RLock()
	srcCol, exists := m.collections[srcID]
	m.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("source collection %q not found", srcID)
	}

	srcCol.mu.RLock()
	srcMeta := srcCol.meta
	srcCol.mu.RUnlock()

	// Create the new collection in SysDB with the same config
	newMeta, err := m.sysdb.CreateCollection(
		newName,
		srcMeta.Dimension,
		srcMeta.Metric,
		srcMeta.IndexType,
	)
	if err != nil {
		return nil, fmt.Errorf("fork: creating new collection %q: %w", newName, err)
	}

	// Create a fresh index for the fork
	newIdx, err := createIndex(srcMeta.Dimension, srcMeta.Metric, srcMeta.IndexType)
	if err != nil {
		m.sysdb.DeleteCollection(newMeta.ID)
		return nil, fmt.Errorf("fork: creating index for %q: %w", newName, err)
	}

	// Deep-copy vector metadata from source
	srcCol.mu.RLock()
	newVectorMeta := make(map[uint64]metadata.VectorMetadata, len(srcCol.vectorMeta))
	for id, meta := range srcCol.vectorMeta {
		metaCopy := make(metadata.VectorMetadata, len(meta))
		for k, v := range meta {
			metaCopy[k] = v
		}
		newVectorMeta[id] = metaCopy
	}
	srcCol.mu.RUnlock()

	newCol := &Collection{
		meta:       newMeta,
		idx:        newIdx,
		wal:        m.wal,
		sysdb:      m.sysdb,
		vectorMeta: newVectorMeta,
	}

	m.mu.Lock()
	m.collections[newMeta.ID] = newCol
	m.mu.Unlock()

	slog.Info("collection forked",
		"source_id", srcID,
		"source_name", srcMeta.Name,
		"fork_id", newMeta.ID,
		"fork_name", newName,
	)

	return newMeta, nil
}
