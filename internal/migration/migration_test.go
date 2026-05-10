package migration

import (
	"fmt"
	"testing"
	"time"
)

func TestJobState_String(t *testing.T) {
	tests := []struct {
		state JobState
		want  string
	}{
		{StatePending, "pending"},
		{StateInProgress, "in_progress"},
		{StateCompleted, "completed"},
		{StateFailed, "failed"},
		{StateCancelled, "cancelled"},
		{JobState(99), "unknown"},
	}
	for _, tt := range tests {
		if got := tt.state.String(); got != tt.want {
			t.Errorf("JobState(%d).String() = %q, want %q", tt.state, got, tt.want)
		}
	}
}

func TestJob_Basic(t *testing.T) {
	job := Job{
		ID:           "mig-001",
		CollectionID: "col-1",
		SourceNode:   "node-1",
		TargetNode:   "node-2",
		State:        StatePending,
		CreatedAt:    time.Now(),
	}

	if job.ID != "mig-001" {
		t.Errorf("ID = %q, want mig-001", job.ID)
	}
	if job.State != StatePending {
		t.Errorf("State = %v, want pending", job.State)
	}
}

func TestPlanner_PlanMigration(t *testing.T) {
	p := NewPlanner(Config{
		BatchSize:   100,
		Parallelism: 4,
	})

	// Plan migration from node-1 to node-2
	plan := p.PlanMigration("col-1", "node-1", "node-2", 1000)

	if plan.CollectionID != "col-1" {
		t.Errorf("CollectionID = %q, want col-1", plan.CollectionID)
	}
	if plan.SourceNode != "node-1" {
		t.Errorf("SourceNode = %q, want node-1", plan.SourceNode)
	}
	if plan.TargetNode != "node-2" {
		t.Errorf("TargetNode = %q, want node-2", plan.TargetNode)
	}
	if plan.TotalVectors != 1000 {
		t.Errorf("TotalVectors = %d, want 1000", plan.TotalVectors)
	}
	if plan.BatchCount != 10 {
		t.Errorf("BatchCount = %d, want 10 (1000/100)", plan.BatchCount)
	}
}

func TestPlanner_PlanMigration_SmallDataset(t *testing.T) {
	p := NewPlanner(Config{
		BatchSize:   100,
		Parallelism: 4,
	})

	plan := p.PlanMigration("col-1", "node-1", "node-2", 50)

	// 50 vectors / 100 batch size = 1 batch (ceiling)
	if plan.BatchCount != 1 {
		t.Errorf("BatchCount = %d, want 1", plan.BatchCount)
	}
}

func TestPlanner_PlanMigration_ZeroVectors(t *testing.T) {
	p := NewPlanner(Config{
		BatchSize:   100,
		Parallelism: 4,
	})

	plan := p.PlanMigration("col-1", "node-1", "node-2", 0)

	if plan.BatchCount != 0 {
		t.Errorf("BatchCount = %d, want 0", plan.BatchCount)
	}
}

func TestWorker_Basic(t *testing.T) {
	w := NewWorker(Config{
		BatchSize:   100,
		Parallelism: 2,
		MaxRetries:  3,
	})

	if w == nil {
		t.Fatal("NewWorker returned nil")
	}
}

func TestWorker_ExecutePlan(t *testing.T) {
	w := NewWorker(Config{
		BatchSize:   10,
		Parallelism: 2,
		MaxRetries:  1,
	})

	// Track transferred batches
	var transferred int
	transferFn := func(batch []uint64) error {
		transferred += len(batch)
		return nil
	}

	plan := MigrationPlan{
		ID:           "mig-001",
		CollectionID: "col-1",
		SourceNode:   "node-1",
		TargetNode:   "node-2",
		TotalVectors: 25,
		BatchCount:   3,
	}

	err := w.ExecutePlan(plan, transferFn)
	if err != nil {
		t.Fatalf("ExecutePlan: %v", err)
	}

	// Should have transferred all 25 vectors
	if transferred != 25 {
		t.Errorf("transferred = %d, want 25", transferred)
	}
}

func TestWorker_ExecutePlan_TransferError(t *testing.T) {
	w := NewWorker(Config{
		BatchSize:   10,
		Parallelism: 1,
		MaxRetries:  2,
	})

	callCount := 0
	transferFn := func(batch []uint64) error {
		callCount++
		if callCount <= 2 {
			return fmt.Errorf("transient error")
		}
		return nil
	}

	plan := MigrationPlan{
		ID:           "mig-002",
		CollectionID: "col-1",
		SourceNode:   "node-1",
		TargetNode:   "node-2",
		TotalVectors: 10,
		BatchCount:   1,
	}

	err := w.ExecutePlan(plan, transferFn)
	// Should succeed after retries
	if err != nil {
		t.Fatalf("ExecutePlan: %v", err)
	}
}

func TestWorker_ExecutePlan_AllRetriesFail(t *testing.T) {
	w := NewWorker(Config{
		BatchSize:   10,
		Parallelism: 1,
		MaxRetries:  2,
	})

	transferFn := func(batch []uint64) error {
		return fmt.Errorf("permanent error")
	}

	plan := MigrationPlan{
		ID:           "mig-003",
		CollectionID: "col-1",
		SourceNode:   "node-1",
		TargetNode:   "node-2",
		TotalVectors: 10,
		BatchCount:   1,
	}

	err := w.ExecutePlan(plan, transferFn)
	if err == nil {
		t.Fatal("expected error after all retries fail")
	}
}

func TestWorker_Progress(t *testing.T) {
	w := NewWorker(Config{
		BatchSize:   10,
		Parallelism: 1,
		MaxRetries:  1,
	})

	var progressUpdates []float64
	w.OnProgress(func(progress float64) {
		progressUpdates = append(progressUpdates, progress)
	})

	transferFn := func(batch []uint64) error {
		return nil
	}

	plan := MigrationPlan{
		ID:           "mig-004",
		CollectionID: "col-1",
		SourceNode:   "node-1",
		TargetNode:   "node-2",
		TotalVectors: 30,
		BatchCount:   3,
	}

	err := w.ExecutePlan(plan, transferFn)
	if err != nil {
		t.Fatalf("ExecutePlan: %v", err)
	}

	// Should have progress updates
	if len(progressUpdates) == 0 {
		t.Error("no progress updates received")
	}

	// Final progress should be 1.0
	last := progressUpdates[len(progressUpdates)-1]
	if last != 1.0 {
		t.Errorf("final progress = %f, want 1.0", last)
	}
}
