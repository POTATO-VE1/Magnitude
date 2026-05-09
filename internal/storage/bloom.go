// Package storage — segment-level bloom filter for fast negative lookups.
//
// Inspired by dbeel's per-SSTable bloom filters (src/storage_engine/lsm_tree.rs:85).
// Before reading a segment file, the bloom filter is checked. If the ID is
// definitely not present, the disk read is skipped entirely.
//
// Bloom filters are built during compaction and stored as sidecar files
// alongside segment data files (e.g., "segment-001.data.bloom").
package storage

import (
	"encoding/binary"
	"fmt"
	"os"

	"github.com/bits-and-blooms/bloom/v3"
)

const (
	// BloomMaxFP is the default false positive rate (1%).
	BloomMaxFP = 0.01

	// BloomMinVectors is the minimum vector count before a bloom filter
	// is created. Below this threshold, the overhead isn't justified.
	BloomMinVectors = 1024
)

// SegmentBloom wraps a bloom filter for fast negative lookups on vector IDs.
type SegmentBloom struct {
	filter *bloom.BloomFilter
	count  uint32
}

// NewSegmentBloom creates a new bloom filter sized for the expected number
// of vectors with the given false positive rate.
func NewSegmentBloom(expectedVectors uint32, falsePositiveRate float64) *SegmentBloom {
	return &SegmentBloom{
		filter: bloom.NewWithEstimates(uint(expectedVectors), falsePositiveRate),
		count:  0,
	}
}

// Add inserts a vector ID into the bloom filter.
func (b *SegmentBloom) Add(id uint64) {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], id)
	b.filter.Add(buf[:])
	b.count++
}

// Test checks whether a vector ID might be in the filter.
// Returns false if the ID is definitely NOT present (true negative).
// Returns true if the ID MIGHT be present (possible false positive).
func (b *SegmentBloom) Test(id uint64) bool {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], id)
	return b.filter.Test(buf[:])
}

// Count returns the number of IDs added to the filter.
func (b *SegmentBloom) Count() uint32 {
	return b.count
}

// WriteToFile serializes the bloom filter to disk.
// Format: [count:4 bytes][bloom data: variable]
func (b *SegmentBloom) WriteToFile(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("bloom: creating %q: %w", path, err)
	}
	defer f.Close()

	// Write count
	if err := binary.Write(f, binary.LittleEndian, b.count); err != nil {
		return fmt.Errorf("bloom: writing count: %w", err)
	}

	// Write bloom filter data
	if _, err := b.filter.WriteTo(f); err != nil {
		return fmt.Errorf("bloom: writing filter data: %w", err)
	}

	if err := f.Sync(); err != nil {
		return fmt.Errorf("bloom: fsync: %w", err)
	}

	return nil
}

// ReadBloomFromFile reads a bloom filter from disk.
func ReadBloomFromFile(path string) (*SegmentBloom, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("bloom: opening %q: %w", path, err)
	}
	defer f.Close()

	// Read count
	var count uint32
	if err := binary.Read(f, binary.LittleEndian, &count); err != nil {
		return nil, fmt.Errorf("bloom: reading count: %w", err)
	}

	// Read bloom filter data
	filter := &bloom.BloomFilter{}
	if _, err := filter.ReadFrom(f); err != nil {
		return nil, fmt.Errorf("bloom: reading filter data: %w", err)
	}

	return &SegmentBloom{
		filter: filter,
		count:  count,
	}, nil
}
