package storage

import (
	"bytes"
	"testing"
)

// FuzzReadHeader fuzzes the binary file header parser to ensure it does not panic.
func FuzzReadHeader(f *testing.F) {
	// Seed corpus with a valid header (magic bytes, version, padding, etc)
	validHeader := make([]byte, 64)
	copy(validHeader[0:4], []byte("VDBX"))
	validHeader[4] = 1 // version
	f.Add(validHeader)
	f.Add(make([]byte, 64)) // zeroed out
	f.Add([]byte("short"))  // too short

	f.Fuzz(func(t *testing.T, data []byte) {
		reader := bytes.NewReader(data)
		// ReadHeader must not panic on malformed input, only return error
		_, _ = ReadHeader(reader)
	})
}
