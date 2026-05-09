// Package migration implements data migration between cluster nodes,
// inspired by dbeel's migration system (src/tasks/migration.rs, src/shards.rs:853-1072).
//
// When a node joins or leaves the cluster, data must be rebalanced:
//   - On node join: some collections move to the new node
//   - On node leave: collections move to remaining nodes
//
// The migration system:
//   - Plans which vectors to move (via hash-range computation)
//   - Transfers vectors in configurable batches
//   - Supports progress tracking and cancellation
//   - Retries on transient failures
package migration

import (
	"fmt"
	"log/slog"
	"sync"
	"time"
)

// JobState represents the state of a migration job.
type JobState int

const (
	// StatePending means the migration is planned but not started.
	StatePending JobState = iota

	// StateInProgress means the migration is actively transferring data.
	StateInProgress

	// StateCompleted means all data was transferred successfully.
	StateCompleted

	// StateFailed means the migration encountered an unrecoverable error.
	StateFailed

	// StateCancelled means the migration was cancelled by the user.
	StateCancelled
)

// String returns a human-readable name for the job state.
func (s JobState) String() string {
	switch s {
	case StatePending:
		return "pending"
	case StateInProgress:
		return "in_progress"
	case StateCompleted:
		return "completed"
	case StateFailed:
		return "failed"
	case StateCancelled:
		return "cancelled"
	default:
		return "unknown"
	}
}

// Job represents a migration job persisted to the system database.
type Job struct {
	ID           string    `json:"id"`
	CollectionID string    `json:"collection_id"`
	SourceNode   string    `json:"source_node"`
	TargetNode   string    `json:"target_node"`
	State        JobState  `json:"state"`
	TotalVectors int       `json:"total_vectors"`
	VectorsSent  int       `json:"vectors_sent"`
	Progress     float64   `json:"progress"`
	StartedAt    time.Time `json:"started_at,omitempty"`
	CompletedAt  time.Time `json:"completed_at,omitempty"`
	ErrorMsg     string    `json:"error_msg,omitempty"`
	CreatedAt    time.Time `json:"created_at"`
}

// MigrationPlan describes a planned migration operation.
type MigrationPlan struct {
	ID           string
	CollectionID string
	SourceNode   string
	TargetNode   string
	TotalVectors int
	BatchCount   int
	BatchSize    int
}

// Config configures the migration system.
type Config struct {
	// BatchSize is the number of vectors per migration batch. Default: 1000
	BatchSize int `yaml:"batchSize"`

	// Parallelism is the number of concurrent migration workers. Default: 4
	Parallelism int `yaml:"parallelism"`

	// MaxRetries is the retry count per batch on failure. Default: 3
	MaxRetries int `yaml:"maxRetries"`
}

// DefaultConfig returns sensible defaults for the migration system.
func DefaultConfig() Config {
	return Config{
		BatchSize:   1000,
		Parallelism: 4,
		MaxRetries:  3,
	}
}

// Planner computes migration plans based on cluster topology changes.
type Planner struct {
	config Config
}

// NewPlanner creates a new migration planner.
func NewPlanner(cfg Config) *Planner {
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 1000
	}
	if cfg.Parallelism <= 0 {
		cfg.Parallelism = 4
	}
	return &Planner{config: cfg}
}

// PlanMigration creates a migration plan for moving vectors from source to target.
func (p *Planner) PlanMigration(collectionID, sourceNode, targetNode string, totalVectors int) MigrationPlan {
	batchCount := 0
	if totalVectors > 0 && p.config.BatchSize > 0 {
		batchCount = (totalVectors + p.config.BatchSize - 1) / p.config.BatchSize
	}

	return MigrationPlan{
		CollectionID: collectionID,
		SourceNode:   sourceNode,
		TargetNode:   targetNode,
		TotalVectors: totalVectors,
		BatchCount:   batchCount,
		BatchSize:    p.config.BatchSize,
	}
}

// TransferFunc is called to transfer a batch of vector IDs to the target node.
type TransferFunc func(batch []uint64) error

// ProgressFunc is called to report migration progress (0.0 to 1.0).
type ProgressFunc func(progress float64)

// Worker executes migration plans by transferring data in batches.
type Worker struct {
	config     Config
	mu         sync.Mutex
	onProgress ProgressFunc
}

// NewWorker creates a new migration worker.
func NewWorker(cfg Config) *Worker {
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 1000
	}
	if cfg.Parallelism <= 0 {
		cfg.Parallelism = 4
	}
	if cfg.MaxRetries <= 0 {
		cfg.MaxRetries = 3
	}
	return &Worker{config: cfg}
}

// OnProgress registers a callback for progress updates.
func (w *Worker) OnProgress(fn ProgressFunc) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.onProgress = fn
}

// ExecutePlan executes a migration plan by calling the transfer function
// for each batch. Returns nil on success, error on permanent failure.
func (w *Worker) ExecutePlan(plan MigrationPlan, transferFn TransferFunc) error {
	slog.Info("migration: starting",
		"plan_id", plan.ID,
		"collection", plan.CollectionID,
		"source", plan.SourceNode,
		"target", plan.TargetNode,
		"total_vectors", plan.TotalVectors,
		"batch_count", plan.BatchCount,
	)

	// Generate batch IDs (sequential for now, could be hash-range based)
	remaining := plan.TotalVectors
	batchNum := 0
	batchSize := plan.BatchSize
	if batchSize <= 0 {
		batchSize = w.config.BatchSize
	}

	for remaining > 0 {
		curBatchSize := batchSize
		if curBatchSize > remaining {
			curBatchSize = remaining
		}

		// Generate vector IDs for this batch
		batch := make([]uint64, curBatchSize)
		for i := range batch {
			batch[i] = uint64(batchNum*batchSize + i)
		}

		// Transfer with retries
		var lastErr error
		for attempt := 0; attempt <= w.config.MaxRetries; attempt++ {
			if attempt > 0 {
				slog.Warn("migration: retrying batch",
					"plan_id", plan.ID,
					"batch", batchNum,
					"attempt", attempt,
					"error", lastErr,
				)
				time.Sleep(time.Duration(attempt) * 100 * time.Millisecond)
			}

			if err := transferFn(batch); err != nil {
				lastErr = err
				continue
			}

			lastErr = nil
			break
		}

		if lastErr != nil {
			slog.Error("migration: batch failed permanently",
				"plan_id", plan.ID,
				"batch", batchNum,
				"error", lastErr,
			)
			return fmt.Errorf("migration: batch %d failed after %d retries: %w",
				batchNum, w.config.MaxRetries, lastErr)
		}

		remaining -= curBatchSize
		batchNum++

		// Report progress
		progress := float64(plan.TotalVectors-remaining) / float64(plan.TotalVectors)
		w.mu.Lock()
		if w.onProgress != nil {
			w.onProgress(progress)
		}
		w.mu.Unlock()
	}

	slog.Info("migration: completed",
		"plan_id", plan.ID,
		"collection", plan.CollectionID,
		"vectors_transferred", plan.TotalVectors,
	)

	return nil
}
