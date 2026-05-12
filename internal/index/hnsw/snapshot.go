package hnsw

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"
)

var snapshotMagic = [4]byte{'H', 'S', 'N', 'P'}

const snapshotVersion uint32 = 1

const snapshotHeaderSize = 64

// SnapshotHeader is the on-disk header for HNSW snapshot files.
// Total size: 64 bytes (fixed, padded for alignment).
type SnapshotHeader struct {
	Magic      [4]byte  // "HSNP"
	Version    uint32   // file format version
	Dim        uint32   // vector dimension
	M          uint32   // max connections per layer
	NodeCount  uint64   // number of nodes
	MaxLevel   int32    // current max level
	EntryPoint int64    // entry point node index (-1 if empty)
	SeqID      uint64   // WAL seqID at snapshot time
	Checksum   [32]byte // SHA-256 of data section
}

// SnapshotToFile serializes the HNSW graph to a binary file using atomic rename.
// The snapshot includes all nodes, their vectors, and the full adjacency lists.
// The seqID parameter records the WAL position at snapshot time for replay coordination.
func (h *HNSWIndex) SnapshotToFile(path string, seqID uint64) error {
	h.mu.RLock()
	defer h.mu.RUnlock()

	// Serialize data section to buffer first (for checksum computation)
	var dataBuf bytes.Buffer

	for _, n := range h.nodes {
		// node id
		if err := binary.Write(&dataBuf, binary.LittleEndian, n.id); err != nil {
			return fmt.Errorf("hnsw snapshot: writing node id: %w", err)
		}
		// level
		if err := binary.Write(&dataBuf, binary.LittleEndian, int32(n.level)); err != nil {
			return fmt.Errorf("hnsw snapshot: writing node level: %w", err)
		}
		// vector
		for _, v := range n.vector {
			if err := binary.Write(&dataBuf, binary.LittleEndian, math.Float32bits(v)); err != nil {
				return fmt.Errorf("hnsw snapshot: writing vector: %w", err)
			}
		}
		// friends per level
		for _, friends := range n.friends {
			count := uint32(len(friends))
			if err := binary.Write(&dataBuf, binary.LittleEndian, count); err != nil {
				return fmt.Errorf("hnsw snapshot: writing friend count: %w", err)
			}
			for _, f := range friends {
				if err := binary.Write(&dataBuf, binary.LittleEndian, int32(f)); err != nil {
					return fmt.Errorf("hnsw snapshot: writing friend index: %w", err)
				}
			}
		}
	}

	// Compute checksum of data section
	checksum := sha256.Sum256(dataBuf.Bytes())

	// Build header
	hdr := SnapshotHeader{
		Magic:      snapshotMagic,
		Version:    snapshotVersion,
		Dim:        uint32(h.dim),
		M:          uint32(h.m),
		NodeCount:  uint64(len(h.nodes)),
		MaxLevel:   int32(h.maxLevel),
		EntryPoint: int64(h.entryPoint),
		SeqID:      seqID,
		Checksum:   checksum,
	}

	// Write to temp file in the same directory for atomic rename
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return fmt.Errorf("hnsw snapshot: creating dir: %w", err)
	}

	tmpFile, err := os.CreateTemp(dir, "hsnp-*.tmp")
	if err != nil {
		return fmt.Errorf("hnsw snapshot: creating temp file: %w", err)
	}
	tmpPath := tmpFile.Name()
	defer func() {
		if tmpFile != nil {
			tmpFile.Close()
			os.Remove(tmpPath)
		}
	}()

	// Write header
	if err := binary.Write(tmpFile, binary.LittleEndian, &hdr); err != nil {
		return fmt.Errorf("hnsw snapshot: writing header: %w", err)
	}

	// Write data section
	if _, err := tmpFile.Write(dataBuf.Bytes()); err != nil {
		return fmt.Errorf("hnsw snapshot: writing data: %w", err)
	}

	// Fsync before rename
	if err := tmpFile.Sync(); err != nil {
		return fmt.Errorf("hnsw snapshot: fsync: %w", err)
	}
	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("hnsw snapshot: close: %w", err)
	}
	tmpFile = nil // prevent deferred cleanup

	// Atomic rename
	if err := os.Rename(tmpPath, path); err != nil {
		return fmt.Errorf("hnsw snapshot: atomic rename: %w", err)
	}

	return nil
}

// LoadHNSWFromSnapshot reads an HNSW graph from a snapshot file.
// Returns the reconstructed index and the WAL seqID recorded at snapshot time.
func LoadHNSWFromSnapshot(path string) (*HNSWIndex, uint64, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, fmt.Errorf("hnsw snapshot: reading file: %w", err)
	}

	if len(raw) < snapshotHeaderSize {
		return nil, 0, fmt.Errorf("hnsw snapshot: file too small (%d bytes)", len(raw))
	}

	// Parse header
	var hdr SnapshotHeader
	if err := binary.Read(bytes.NewReader(raw[:snapshotHeaderSize]), binary.LittleEndian, &hdr); err != nil {
		return nil, 0, fmt.Errorf("hnsw snapshot: reading header: %w", err)
	}
	if hdr.Magic != snapshotMagic {
		return nil, 0, fmt.Errorf("hnsw snapshot: invalid magic bytes: %v", hdr.Magic)
	}
	if hdr.Version != snapshotVersion {
		return nil, 0, fmt.Errorf("hnsw snapshot: unsupported version %d", hdr.Version)
	}

	dataBytes := raw[snapshotHeaderSize:]
	checksum := sha256.Sum256(dataBytes)
	if checksum != hdr.Checksum {
		return nil, 0, fmt.Errorf("hnsw snapshot: checksum mismatch")
	}

	// Parse nodes
	r := bytes.NewReader(dataBytes)
	dim := int(hdr.Dim)
	m := int(hdr.M)
	nodes := make([]node, hdr.NodeCount)
	idToNode := make(map[uint64]int, hdr.NodeCount)

	for i := 0; i < int(hdr.NodeCount); i++ {
		var id uint64
		var level int32
		if err := binary.Read(r, binary.LittleEndian, &id); err != nil {
			return nil, 0, fmt.Errorf("hnsw snapshot: reading node %d id: %w", i, err)
		}
		if err := binary.Read(r, binary.LittleEndian, &level); err != nil {
			return nil, 0, fmt.Errorf("hnsw snapshot: reading node %d level: %w", i, err)
		}

		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			var bits uint32
			if err := binary.Read(r, binary.LittleEndian, &bits); err != nil {
				return nil, 0, fmt.Errorf("hnsw snapshot: reading node %d vector[%d]: %w", i, j, err)
			}
			vec[j] = math.Float32frombits(bits)
		}

		friends := make([][]int, int(level)+1)
		for lc := 0; lc <= int(level); lc++ {
			var count uint32
			if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
				return nil, 0, fmt.Errorf("hnsw snapshot: reading node %d level %d friend count: %w", i, lc, err)
			}
			layerFriends := make([]int, count)
			for j := 0; j < int(count); j++ {
				var idx int32
				if err := binary.Read(r, binary.LittleEndian, &idx); err != nil {
					return nil, 0, fmt.Errorf("hnsw snapshot: reading node %d level %d friend %d: %w", i, lc, j, err)
				}
				layerFriends[j] = int(idx)
			}
			friends[lc] = layerFriends
		}

		nodes[i] = node{
			id:      id,
			vector:  vec,
			level:   int(level),
			friends: friends,
		}
		idToNode[id] = i
	}

	// Reconstruct HNSWIndex
	h := &HNSWIndex{
		dim:            dim,
		metric:         "l2", // default; will be overridden by caller if needed
		m:              m,
		mMax0:          2 * m,
		efConstruction: 200,
		efSearch:       50,
		mL:             1.0 / math.Log(float64(m)),
		maxLevel:       int(hdr.MaxLevel),
		entryPoint:     int(hdr.EntryPoint),
		nodes:          nodes,
		idToNode:       idToNode,
		rng:            rand.New(rand.NewSource(time.Now().UnixNano())),
		deleted:        make(map[int]bool),
		visited:        make([]uint64, len(nodes)),
	}

	return h, hdr.SeqID, nil
}
