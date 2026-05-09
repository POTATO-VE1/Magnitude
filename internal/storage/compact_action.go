// Package storage — crash-safe compaction action files.
//
// Inspired by dbeel's CompactionAction (src/storage_engine/lsm_tree.rs:73-77).
// Before performing renames/deletes during compaction, we write an action file
// to disk. On startup, any incomplete actions are replayed. This ensures
// multi-file atomicity: either all renames/deletes happen, or none do.
//
// The action file is deleted after successful execution, so only incomplete
// actions survive a crash.
package storage

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// CompactionActionFileExt is the file extension for compaction action manifests.
const CompactionActionFileExt = ".compact_action"

// FileRename describes a single rename operation within a compaction action.
type FileRename struct {
	Source      string `json:"source"`
	Destination string `json:"destination"`
}

// CompactionAction describes the set of renames and deletes that constitute
// an atomic compaction. Written to disk BEFORE execution so that a crash
// mid-operation can be recovered.
type CompactionAction struct {
	Renames []FileRename `json:"renames"`
	Deletes []string     `json:"deletes"`
}

// CompactionActionPath returns the path for the action file with the given index.
func CompactionActionPath(dir string, index int) string {
	return filepath.Join(dir, fmt.Sprintf("%020d%s", index, CompactionActionFileExt))
}

// WriteCompactionAction serializes the action to JSON and writes it to disk
// with an fsync for durability.
func WriteCompactionAction(path string, action *CompactionAction) error {
	data, err := json.MarshalIndent(action, "", "  ")
	if err != nil {
		return fmt.Errorf("compaction action: marshal: %w", err)
	}

	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("compaction action: write %q: %w", path, err)
	}

	// Fsync the file to ensure durability before we start executing renames
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("compaction action: open for fsync: %w", err)
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return fmt.Errorf("compaction action: fsync: %w", err)
	}
	f.Close()

	return nil
}

// ReadCompactionAction reads and deserializes a compaction action file.
func ReadCompactionAction(path string) (*CompactionAction, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("compaction action: read %q: %w", path, err)
	}

	var action CompactionAction
	if err := json.Unmarshal(data, &action); err != nil {
		return nil, fmt.Errorf("compaction action: unmarshal %q: %w", path, err)
	}

	return &action, nil
}

// ExecuteCompactionAction performs all renames then all deletes.
// Each operation is idempotent: missing source files are skipped gracefully.
// This is critical for crash recovery — a partial execution can be replayed.
func ExecuteCompactionAction(action *CompactionAction) error {
	// Phase 1: Renames (idempotent — skip if source missing)
	for _, r := range action.Renames {
		if _, err := os.Stat(r.Source); os.IsNotExist(err) {
			slog.Debug("compaction action: rename source missing, skipping",
				"source", r.Source,
				"destination", r.Destination,
			)
			continue
		}
		if err := os.Rename(r.Source, r.Destination); err != nil {
			return fmt.Errorf("compaction action: rename %q → %q: %w", r.Source, r.Destination, err)
		}
		slog.Debug("compaction action: renamed", "source", r.Source, "destination", r.Destination)
	}

	// Phase 2: Deletes (idempotent — skip if file missing)
	for _, d := range action.Deletes {
		if _, err := os.Stat(d); os.IsNotExist(err) {
			slog.Debug("compaction action: delete target missing, skipping", "path", d)
			continue
		}
		if err := os.Remove(d); err != nil {
			return fmt.Errorf("compaction action: delete %q: %w", d, err)
		}
		slog.Debug("compaction action: deleted", "path", d)
	}

	return nil
}

// RecoverCompactionActions scans a directory for .compact_action files,
// replays each one, and deletes the action file after successful execution.
// Called once during startup before WAL replay.
func RecoverCompactionActions(dir string) error {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("compaction action: reading dir %q: %w", dir, err)
	}

	// Sort by filename to replay in order
	var actionPaths []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), CompactionActionFileExt) {
			actionPaths = append(actionPaths, filepath.Join(dir, e.Name()))
		}
	}
	sort.Strings(actionPaths)

	if len(actionPaths) == 0 {
		return nil
	}

	slog.Info("compaction action: recovering incomplete actions", "count", len(actionPaths))

	for _, path := range actionPaths {
		action, err := ReadCompactionAction(path)
		if err != nil {
			slog.Error("compaction action: failed to read, skipping",
				"path", path,
				"error", err,
			)
			continue
		}

		if err := ExecuteCompactionAction(action); err != nil {
			slog.Error("compaction action: failed to execute",
				"path", path,
				"error", err,
			)
			continue
		}

		// Remove the action file after successful execution
		if err := os.Remove(path); err != nil {
			slog.Warn("compaction action: failed to remove action file",
				"path", path,
				"error", err,
			)
		} else {
			slog.Info("compaction action: recovered and cleaned up", "path", path)
		}
	}

	return nil
}
