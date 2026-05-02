// Package storage — POSIX memory-mapped file operations.
//
// Memory-mapping (mmap) allows the OS to page vector data in/out of RAM on demand,
// making the effective index size limited by disk (not RAM). The OS page cache acts
// as a transparent LRU cache for recently accessed vector blocks.
//
// Critical implementation note from the spec:
//   Use unsafe.Slice (Go 1.17+) to cast mmap'd []byte to []float32.
//   NEVER use reflect.SliceHeader — the GC can move the underlying mmap page table
//   entry between the uintptr computation and the slice header write, yielding a
//   dangling base address and silent memory corruption.
//
// Build constraint: //go:build !windows
// Windows implementation lives in mmap_windows.go.

//go:build !windows

package storage

import (
	"fmt"
	"os"
	"unsafe"

	"golang.org/x/sys/unix"
)

// MappedFile provides zero-copy access to vectors stored in a memory-mapped file.
// The underlying []byte is mapped with MAP_SHARED so writes are visible to the OS
// page cache and will be written back to disk.
type MappedFile struct {
	file    *os.File
	data    []byte    // mmap'd region
	Vectors []float32 // zero-copy float32 view into data (past the header)
	Size    int64     // total file size in bytes
	Count   uint64    // number of vectors
	Dim     int       // vector dimension
}

// OpenMapped opens or creates a memory-mapped vector file.
// If the file exists, its header is validated and the data section is mapped.
// If the file does not exist, a new file is created with capacity for n vectors
// of the given dimension.
func OpenMapped(path string, n int, dim int) (*MappedFile, error) {
	fileSize := int64(headerSize) + int64(n)*int64(dim)*4
	if fileSize <= 0 {
		return nil, fmt.Errorf("storage: invalid file size: n=%d dim=%d", n, dim)
	}

	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return nil, fmt.Errorf("storage: opening %q: %w", path, err)
	}

	// Stat to check existing size
	fi, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("storage: stat %q: %w", path, err)
	}

	if fi.Size() < fileSize {
		// Extend file to target size
		if err := f.Truncate(fileSize); err != nil {
			f.Close()
			return nil, fmt.Errorf("storage: truncate %q to %d: %w", path, fileSize, err)
		}
	} else {
		fileSize = fi.Size()
	}

	// Memory-map the file with MAP_SHARED (writes propagate to disk)
	data, err := unix.Mmap(int(f.Fd()), 0, int(fileSize),
		unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("storage: mmap %q: %w", path, err)
	}

	// Create zero-copy float32 view of the data section (past the 64-byte header).
	// unsafe.Slice is safe here because:
	//   1. The mmap region is pinned by the kernel — GC cannot move it.
	//   2. The pointer is derived from the mmap'd []byte, which is a stable address.
	dataSection := data[headerSize:]
	floatCount := len(dataSection) / 4
	var vectors []float32
	if floatCount > 0 {
		vectors = unsafe.Slice((*float32)(unsafe.Pointer(&dataSection[0])), floatCount)
	}

	return &MappedFile{
		file:    f,
		data:    data,
		Vectors: vectors,
		Size:    fileSize,
		Count:   uint64(n),
		Dim:     dim,
	}, nil
}

// Flush synchronously writes all modified pages to disk via msync.
func (mf *MappedFile) Flush() error {
	if err := unix.Msync(mf.data, unix.MS_SYNC); err != nil {
		return fmt.Errorf("storage: msync: %w", err)
	}
	return nil
}

// Close flushes, unmaps, and closes the underlying file.
func (mf *MappedFile) Close() error {
	if err := mf.Flush(); err != nil {
		return err
	}
	if err := unix.Munmap(mf.data); err != nil {
		return fmt.Errorf("storage: munmap: %w", err)
	}
	return mf.file.Close()
}

// AdviseSequential tells the OS to prefetch pages sequentially.
// Best for full-scan operations (brute-force search, compaction).
func (mf *MappedFile) AdviseSequential() error {
	return unix.Madvise(mf.data, unix.MADV_SEQUENTIAL)
}

// AdviseRandom tells the OS to disable prefetch.
// Best for random access patterns (IVF posting list lookup).
func (mf *MappedFile) AdviseRandom() error {
	return unix.Madvise(mf.data, unix.MADV_RANDOM)
}

// GetVector returns a slice into the mmap'd vectors at the given row.
// This is zero-copy — the slice points directly into kernel page cache.
// The returned slice must NOT be held across Flush/Close calls.
func (mf *MappedFile) GetVector(row int) []float32 {
	start := row * mf.Dim
	return mf.Vectors[start : start+mf.Dim]
}

// PutVector copies a vector into the mmap'd region at the given row.
func (mf *MappedFile) PutVector(row int, vector []float32) {
	start := row * mf.Dim
	copy(mf.Vectors[start:start+mf.Dim], vector)
}

// Raw returns the underlying mmap'd byte slice (header + data).
func (mf *MappedFile) Raw() []byte {
	return mf.data
}
