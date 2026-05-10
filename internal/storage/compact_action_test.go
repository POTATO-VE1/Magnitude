package storage

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCompactionAction_WriteAndRead(t *testing.T) {
	dir := t.TempDir()

	action := &CompactionAction{
		Renames: []FileRename{
			{Source: filepath.Join(dir, "tmp.data"), Destination: filepath.Join(dir, "seg-001.data")},
			{Source: filepath.Join(dir, "tmp.bloom"), Destination: filepath.Join(dir, "seg-001.bloom")},
		},
		Deletes: []string{
			filepath.Join(dir, "old-seg-000.data"),
			filepath.Join(dir, "old-seg-000.bloom"),
		},
	}

	path := CompactionActionPath(dir, 1)
	if err := WriteCompactionAction(path, action); err != nil {
		t.Fatalf("WriteCompactionAction: %v", err)
	}

	// File should exist
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("action file not created: %v", err)
	}

	// Read it back
	got, err := ReadCompactionAction(path)
	if err != nil {
		t.Fatalf("ReadCompactionAction: %v", err)
	}

	if len(got.Renames) != 2 {
		t.Fatalf("expected 2 renames, got %d", len(got.Renames))
	}
	if got.Renames[0].Source != action.Renames[0].Source {
		t.Errorf("rename source mismatch: got %q, want %q", got.Renames[0].Source, action.Renames[0].Source)
	}
	if got.Renames[1].Destination != action.Renames[1].Destination {
		t.Errorf("rename dest mismatch: got %q, want %q", got.Renames[1].Destination, action.Renames[1].Destination)
	}
	if len(got.Deletes) != 2 {
		t.Fatalf("expected 2 deletes, got %d", len(got.Deletes))
	}
	if got.Deletes[1] != action.Deletes[1] {
		t.Errorf("delete mismatch: got %q, want %q", got.Deletes[1], action.Deletes[1])
	}
}

func TestCompactionAction_ExecuteRenamesAndDeletes(t *testing.T) {
	dir := t.TempDir()

	// Create source files for renames
	src1 := filepath.Join(dir, "tmp.data")
	src2 := filepath.Join(dir, "tmp.bloom")
	dst1 := filepath.Join(dir, "seg-001.data")
	dst2 := filepath.Join(dir, "seg-001.bloom")

	if err := os.WriteFile(src1, []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(src2, []byte("bloom"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Create files to delete
	del1 := filepath.Join(dir, "old-seg-000.data")
	if err := os.WriteFile(del1, []byte("old"), 0o644); err != nil {
		t.Fatal(err)
	}

	action := &CompactionAction{
		Renames: []FileRename{
			{Source: src1, Destination: dst1},
			{Source: src2, Destination: dst2},
		},
		Deletes: []string{del1},
	}

	if err := ExecuteCompactionAction(action); err != nil {
		t.Fatalf("ExecuteCompactionAction: %v", err)
	}

	// Verify renames happened
	if _, err := os.Stat(dst1); err != nil {
		t.Errorf("renamed file dst1 missing: %v", err)
	}
	if _, err := os.Stat(dst2); err != nil {
		t.Errorf("renamed file dst2 missing: %v", err)
	}
	if _, err := os.Stat(src1); !os.IsNotExist(err) {
		t.Errorf("source file src1 should be gone after rename")
	}

	// Verify delete happened
	if _, err := os.Stat(del1); !os.IsNotExist(err) {
		t.Errorf("deleted file del1 should be gone")
	}
}

func TestCompactionAction_ExecuteSkipsMissingSource(t *testing.T) {
	dir := t.TempDir()

	action := &CompactionAction{
		Renames: []FileRename{
			{Source: filepath.Join(dir, "nonexistent.data"), Destination: filepath.Join(dir, "dst.data")},
		},
		Deletes: []string{filepath.Join(dir, "nonexistent-delete.data")},
	}

	// Should not error — idempotent
	if err := ExecuteCompactionAction(action); err != nil {
		t.Fatalf("ExecuteCompactionAction should skip missing files, got: %v", err)
	}
}

func TestCompactionAction_RecoverCompactionActions(t *testing.T) {
	dir := t.TempDir()

	// Create source files
	srcData := filepath.Join(dir, "tmp-recover.data")
	dstData := filepath.Join(dir, "recovered.data")
	if err := os.WriteFile(srcData, []byte("recovered"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Write an action file
	action := &CompactionAction{
		Renames: []FileRename{
			{Source: srcData, Destination: dstData},
		},
		Deletes: []string{},
	}
	actionPath := CompactionActionPath(dir, 1)
	if err := WriteCompactionAction(actionPath, action); err != nil {
		t.Fatal(err)
	}

	// Recover — should replay the action
	if err := RecoverCompactionActions(dir); err != nil {
		t.Fatalf("RecoverCompactionActions: %v", err)
	}

	// Verify rename happened
	if _, err := os.Stat(dstData); err != nil {
		t.Errorf("recovered file missing: %v", err)
	}

	// Verify action file was cleaned up
	if _, err := os.Stat(actionPath); !os.IsNotExist(err) {
		t.Errorf("action file should be deleted after recovery")
	}
}

func TestCompactionAction_RecoverNoActions(t *testing.T) {
	dir := t.TempDir()

	// No action files — should be a no-op
	if err := RecoverCompactionActions(dir); err != nil {
		t.Fatalf("RecoverCompactionActions on empty dir: %v", err)
	}
}

func TestCompactionAction_RecoverIdempotent(t *testing.T) {
	dir := t.TempDir()

	// Create source file
	src := filepath.Join(dir, "tmp-idem.data")
	dst := filepath.Join(dir, "idem.data")
	if err := os.WriteFile(src, []byte("idempotent"), 0o644); err != nil {
		t.Fatal(err)
	}

	action := &CompactionAction{
		Renames: []FileRename{{Source: src, Destination: dst}},
		Deletes: []string{},
	}
	actionPath := CompactionActionPath(dir, 1)
	if err := WriteCompactionAction(actionPath, action); err != nil {
		t.Fatal(err)
	}

	// First recovery — does the rename
	if err := RecoverCompactionActions(dir); err != nil {
		t.Fatalf("first recovery: %v", err)
	}

	// Second recovery — no action files left, should be no-op
	if err := RecoverCompactionActions(dir); err != nil {
		t.Fatalf("second recovery: %v", err)
	}
}

func TestCompactionActionPath(t *testing.T) {
	path := CompactionActionPath("/data", 42)
	expected := filepath.Join("/data", "00000000000000000042.compact_action")
	if path != expected {
		t.Errorf("CompactionActionPath = %q, want %q", path, expected)
	}
}
