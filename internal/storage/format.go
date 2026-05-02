// Package storage — binary file format for vector data files.
//
// File layout:
//   [FileHeader: 64 bytes] [vector data: N * dim * 4 bytes]
//
// FileHeader is fixed-size, little-endian encoded.
// Checksum covers the entire data section (not the header itself).
package storage

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// Magic bytes identifying a VectorDB data file.
var fileMagic = [4]byte{'V', 'D', 'B', 'X'}

const (
	fileVersion    uint16 = 1
	headerSize            = 64
	dataTypeFloat32 uint8 = 1
)

// FileHeader is the on-disk header for vector data files.
// Total size: 64 bytes (fixed, padded for alignment).
type FileHeader struct {
	Magic      [4]byte   // "VDBX"
	Version    uint16    // file format version
	Flags      uint16    // reserved flags
	VectorCount uint64   // number of vectors in the file
	Dimension  uint32    // vector dimension
	DataType   uint8     // 1 = float32
	_          [3]byte   // padding for alignment
	Checksum   [32]byte  // SHA-256 of the data section
	_          [8]byte   // reserved for future use
}

// WriteHeader writes a FileHeader to w in little-endian format.
func WriteHeader(w io.Writer, vectorCount uint64, dim uint32, checksum [32]byte) error {
	hdr := FileHeader{
		Magic:       fileMagic,
		Version:     fileVersion,
		VectorCount: vectorCount,
		Dimension:   dim,
		DataType:    dataTypeFloat32,
		Checksum:    checksum,
	}

	return binary.Write(w, binary.LittleEndian, &hdr)
}

// ReadHeader reads a FileHeader from r.
func ReadHeader(r io.Reader) (*FileHeader, error) {
	var hdr FileHeader
	if err := binary.Read(r, binary.LittleEndian, &hdr); err != nil {
		return nil, fmt.Errorf("storage: reading header: %w", err)
	}
	if hdr.Magic != fileMagic {
		return nil, fmt.Errorf("storage: invalid magic bytes: %v", hdr.Magic)
	}
	if hdr.Version != fileVersion {
		return nil, fmt.Errorf("storage: unsupported file version %d (expected %d)", hdr.Version, fileVersion)
	}
	if hdr.DataType != dataTypeFloat32 {
		return nil, fmt.Errorf("storage: unsupported data type %d", hdr.DataType)
	}
	return &hdr, nil
}

// ComputeChecksum computes the SHA-256 checksum of the data section of a file.
func ComputeChecksum(path string) ([32]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return [32]byte{}, fmt.Errorf("storage: opening file for checksum: %w", err)
	}
	defer f.Close()

	// Seek past the header
	if _, err := f.Seek(headerSize, io.SeekStart); err != nil {
		return [32]byte{}, fmt.Errorf("storage: seeking past header: %w", err)
	}

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return [32]byte{}, fmt.Errorf("storage: computing checksum: %w", err)
	}

	var checksum [32]byte
	copy(checksum[:], h.Sum(nil))
	return checksum, nil
}

// ComputeChecksumFromBytes computes SHA-256 of raw byte data.
func ComputeChecksumFromBytes(data []byte) [32]byte {
	return sha256.Sum256(data)
}

// VerifyFileIntegrity reads a file, validates its header, and verifies the
// SHA-256 checksum of the data section.
func VerifyFileIntegrity(path string) (*FileHeader, error) {
	hdr, err := ReadHeaderFromFile(path)
	if err != nil {
		return nil, err
	}

	actual, err := ComputeChecksum(path)
	if err != nil {
		return nil, err
	}

	if actual != hdr.Checksum {
		return nil, fmt.Errorf("storage: checksum mismatch for %q: expected %x, got %x",
			path, hdr.Checksum, actual)
	}

	return hdr, nil
}

// ReadHeaderFromFile opens a file and reads its header.
func ReadHeaderFromFile(path string) (*FileHeader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("storage: opening %q: %w", path, err)
	}
	defer f.Close()
	return ReadHeader(f)
}
