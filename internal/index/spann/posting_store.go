package spann

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"unsafe"

	"golang.org/x/sys/unix"
)

// PostingStore defines an interface for reading posting lists.
type PostingStore interface {
	GetPostings(centroidID int) ([]posting, error)
	Close() error
}

// MmapPostingStore implements PostingStore over a memory-mapped file.
type MmapPostingStore struct {
	file *os.File
	data []byte
	dim  int

	// Header tracks offsets: centroidID -> (offset, count)
	offsets map[int]struct {
		offset int64
		count  int
	}
}

// NewMmapPostingStore opens an existing postings file.
func NewMmapPostingStore(path string, dim int) (*MmapPostingStore, error) {
	f, err := os.OpenFile(path, os.O_RDONLY, 0600)
	if err != nil {
		return nil, fmt.Errorf("spann: opening posting store %q: %w", path, err)
	}

	fi, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("spann: stat posting store %q: %w", path, err)
	}

	if fi.Size() == 0 {
		return &MmapPostingStore{
			file:    f,
			data:    nil,
			dim:     dim,
			offsets: make(map[int]struct{ offset int64; count int }),
		}, nil
	}

	data, err := unix.Mmap(int(f.Fd()), 0, int(fi.Size()), unix.PROT_READ, unix.MAP_SHARED)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("spann: mmap posting store %q: %w", path, err)
	}
	
	// Advise OS for random access since we read posting lists randomly based on search
	unix.Madvise(data, unix.MADV_RANDOM)

	store := &MmapPostingStore{
		file:    f,
		data:    data,
		dim:     dim,
		offsets: make(map[int]struct{ offset int64; count int }),
	}

	if err := store.readHeader(); err != nil {
		unix.Munmap(data)
		f.Close()
		return nil, err
	}

	return store, nil
}

// readHeader parses the footer/header to find offsets.
// Format: 
// [postings data]
// [centroid_id uint32][offset uint64][count uint32] (repeated)
// [num_centroids uint32]
func (s *MmapPostingStore) readHeader() error {
	data := s.data
	if len(data) < 4 {
		return nil // Empty
	}
	
	numCents := binary.LittleEndian.Uint32(data[len(data)-4:])
	headerLen := int(numCents) * 16 // 4 + 8 + 4
	
	if len(data) < headerLen+4 {
		return fmt.Errorf("invalid posting store file size")
	}

	headerStart := len(data) - 4 - headerLen
	hdata := data[headerStart : len(data)-4]

	for i := 0; i < int(numCents); i++ {
		cid := binary.LittleEndian.Uint32(hdata[i*16 : i*16+4])
		off := binary.LittleEndian.Uint64(hdata[i*16+4 : i*16+12])
		cnt := binary.LittleEndian.Uint32(hdata[i*16+12 : i*16+16])
		s.offsets[int(cid)] = struct { offset int64; count int }{int64(off), int(cnt)}
	}
	return nil
}

// GetPostings returns the posting list for a centroid.
// Returned slices own their memory — safe to hold after the store is closed.
func (s *MmapPostingStore) GetPostings(centroidID int) ([]posting, error) {
	if s.data == nil {
		return nil, nil
	}

	meta, ok := s.offsets[centroidID]
	if !ok || meta.count == 0 {
		return nil, nil
	}

	data := s.data
	vecBytes := s.dim * 4
	recordSize := 8 + vecBytes
	end := meta.offset + int64(meta.count)*int64(recordSize)
	if end > int64(len(data)) {
		return nil, fmt.Errorf("posting list bounds out of range")
	}

	chunk := data[meta.offset:end]
	postings := make([]posting, meta.count)

	for i := 0; i < meta.count; i++ {
		postings[i].id = binary.LittleEndian.Uint64(chunk[i*recordSize : i*recordSize+8])

		// Copy vector data out of mmap into an owned Go slice.
		// This prevents dangling pointers when the mmap store is closed.
		vec := make([]float32, s.dim)
		vecStart := i*recordSize + 8
		for d := 0; d < s.dim; d++ {
			vec[d] = math.Float32frombits(binary.LittleEndian.Uint32(chunk[vecStart+d*4 : vecStart+d*4+4]))
		}
		postings[i].vector = vec
	}

	return postings, nil
}

// Close unmaps the memory and closes the file.
func (s *MmapPostingStore) Close() error {
	var err1 error
	if s.data != nil {
		err1 = unix.Munmap(s.data)
	}
	err2 := s.file.Close()
	if err1 != nil {
		return err1
	}
	return err2
}

// WritePostings writes an array of posting lists to disk and returns the path.
func WritePostings(path string, postings [][]posting, dim int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	var offset int64
	var header []byte

	// Write postings data
	for i, list := range postings {
		if len(list) == 0 {
			continue
		}

		startOff := offset
		for _, p := range list {
			var idBuf [8]byte
			binary.LittleEndian.PutUint64(idBuf[:], p.id)
			f.Write(idBuf[:])
			
			vecData := unsafe.Slice((*byte)(unsafe.Pointer(&p.vector[0])), dim*4)
			f.Write(vecData)
			offset += int64(8 + dim*4)
		}

		// Save header info: centroidID(4), offset(8), count(4)
		var hBuf [16]byte
		binary.LittleEndian.PutUint32(hBuf[0:4], uint32(i))
		binary.LittleEndian.PutUint64(hBuf[4:12], uint64(startOff))
		binary.LittleEndian.PutUint32(hBuf[12:16], uint32(len(list)))
		header = append(header, hBuf[:]...)
	}

	// Write header
	f.Write(header)
	
	// Write numCentroids
	var numBuf [4]byte
	binary.LittleEndian.PutUint32(numBuf[:], uint32(len(header)/16))
	f.Write(numBuf[:])

	// Sync to ensure durability
	return f.Sync()
}
