package storage

import (
	"path/filepath"
	"testing"
)

func TestSegmentBloom_AddAndTest(t *testing.T) {
	bloom := NewSegmentBloom(1000, 0.01)

	// Add some IDs
	bloom.Add(42)
	bloom.Add(100)
	bloom.Add(9999)

	// All added IDs should test positive
	if !bloom.Test(42) {
		t.Error("expected bloom to contain 42")
	}
	if !bloom.Test(100) {
		t.Error("expected bloom to contain 100")
	}
	if !bloom.Test(9999) {
		t.Error("expected bloom to contain 9999")
	}
}

func TestSegmentBloom_AbsentIDs(t *testing.T) {
	bloom := NewSegmentBloom(1000, 0.01)

	bloom.Add(42)
	bloom.Add(100)

	// Test 1000 absent IDs — should have ~1% false positives (≤20 to be safe)
	falsePositives := 0
	for i := uint64(10000); i < 11000; i++ {
		if bloom.Test(i) {
			falsePositives++
		}
	}

	if falsePositives > 30 {
		t.Errorf("too many false positives: %d out of 1000 (expected ≤30)", falsePositives)
	}
}

func TestSegmentBloom_Count(t *testing.T) {
	bloom := NewSegmentBloom(1000, 0.01)

	if bloom.Count() != 0 {
		t.Errorf("expected count 0, got %d", bloom.Count())
	}

	bloom.Add(1)
	bloom.Add(2)
	bloom.Add(3)

	if bloom.Count() != 3 {
		t.Errorf("expected count 3, got %d", bloom.Count())
	}
}

func TestSegmentBloom_WriteAndRead(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.bloom")

	bloom := NewSegmentBloom(1000, 0.01)
	bloom.Add(42)
	bloom.Add(100)
	bloom.Add(9999)

	if err := bloom.WriteToFile(path); err != nil {
		t.Fatalf("WriteToFile: %v", err)
	}

	// Read it back
	loaded, err := ReadBloomFromFile(path)
	if err != nil {
		t.Fatalf("ReadBloomFromFile: %v", err)
	}

	// All original IDs should still test positive
	if !loaded.Test(42) {
		t.Error("loaded bloom should contain 42")
	}
	if !loaded.Test(100) {
		t.Error("loaded bloom should contain 100")
	}
	if !loaded.Test(9999) {
		t.Error("loaded bloom should contain 9999")
	}

	// Absent IDs should still test negative (mostly)
	falsePositives := 0
	for i := uint64(20000); i < 21000; i++ {
		if loaded.Test(i) {
			falsePositives++
		}
	}
	if falsePositives > 30 {
		t.Errorf("loaded bloom: too many false positives: %d", falsePositives)
	}

	// Count should match
	if loaded.Count() != 3 {
		t.Errorf("loaded bloom count: got %d, want 3", loaded.Count())
	}
}

func TestSegmentBloom_ReadFromFile_NotFound(t *testing.T) {
	_, err := ReadBloomFromFile("/nonexistent/path.bloom")
	if err == nil {
		t.Fatal("expected error reading nonexistent file")
	}
}

func TestSegmentBloom_LargeDataset(t *testing.T) {
	bloom := NewSegmentBloom(100000, 0.01)

	// Add 100K IDs
	for i := uint64(0); i < 100000; i++ {
		bloom.Add(i)
	}

	// All should be present
	for i := uint64(0); i < 100000; i++ {
		if !bloom.Test(i) {
			t.Errorf("bloom should contain %d", i)
			break
		}
	}

	if bloom.Count() != 100000 {
		t.Errorf("expected count 100000, got %d", bloom.Count())
	}
}

func TestSegmentBloom_WriteReadRoundTrip_Large(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "large.bloom")

	bloom := NewSegmentBloom(10000, 0.01)
	for i := uint64(0); i < 10000; i++ {
		bloom.Add(i * 7) // sparse IDs
	}

	if err := bloom.WriteToFile(path); err != nil {
		t.Fatal(err)
	}

	loaded, err := ReadBloomFromFile(path)
	if err != nil {
		t.Fatal(err)
	}

	// Verify all added IDs
	for i := uint64(0); i < 10000; i++ {
		id := i * 7
		if !loaded.Test(id) {
			t.Errorf("loaded bloom should contain %d", id)
			break
		}
	}

	if loaded.Count() != 10000 {
		t.Errorf("loaded count: got %d, want 10000", loaded.Count())
	}
}
