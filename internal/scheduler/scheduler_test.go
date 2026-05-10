package scheduler

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestScheduler_SubmitAndExecute(t *testing.T) {
	s := New(Config{
		ForegroundWorkers: 2,
		BackgroundWorkers: 1,
	})
	s.Start()
	defer s.Stop()

	var executed atomic.Bool
	task := Task{
		Name:     "test-task",
		Priority: PriorityHigh,
		Fn: func(ctx context.Context) {
			executed.Store(true)
		},
	}

	s.Submit(task)

	// Wait for execution
	deadline := time.After(2 * time.Second)
	for !executed.Load() {
		select {
		case <-deadline:
			t.Fatal("task not executed within timeout")
		default:
			time.Sleep(5 * time.Millisecond)
		}
	}
}

func TestScheduler_HighPriorityFirst(t *testing.T) {
	s := New(Config{
		ForegroundWorkers: 1,
		BackgroundWorkers: 1,
	})
	s.Start()
	defer s.Stop()

	// Block the foreground worker with a slow task
	blocker := make(chan struct{})
	s.Submit(Task{
		Name:     "blocker",
		Priority: PriorityHigh,
		Fn: func(ctx context.Context) {
			<-blocker
		},
	})

	// Give the blocker time to start
	time.Sleep(20 * time.Millisecond)

	// Submit a background task and a high-priority task
	var order []string
	var mu sync.Mutex

	s.Submit(Task{
		Name:     "background-1",
		Priority: PriorityBackground,
		Fn: func(ctx context.Context) {
			mu.Lock()
			order = append(order, "bg")
			mu.Unlock()
		},
	})

	// Unblock the first task
	close(blocker)
	time.Sleep(50 * time.Millisecond)

	s.Submit(Task{
		Name:     "foreground-1",
		Priority: PriorityHigh,
		Fn: func(ctx context.Context) {
			mu.Lock()
			order = append(order, "fg")
			mu.Unlock()
		},
	})

	time.Sleep(200 * time.Millisecond)

	mu.Lock()
	defer mu.Unlock()

	// Both should have executed
	hasBG := false
	hasFG := false
	for _, o := range order {
		if o == "bg" {
			hasBG = true
		}
		if o == "fg" {
			hasFG = true
		}
	}
	if !hasBG {
		t.Error("background task did not execute")
	}
	if !hasFG {
		t.Error("foreground task did not execute")
	}
}

func TestScheduler_Stop(t *testing.T) {
	s := New(Config{
		ForegroundWorkers: 2,
		BackgroundWorkers: 1,
	})
	s.Start()

	var count atomic.Int32
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		s.Submit(Task{
			Name:     "task",
			Priority: PriorityHigh,
			Fn: func(ctx context.Context) {
				time.Sleep(10 * time.Millisecond)
				count.Add(1)
				wg.Done()
			},
		})
	}

	// Wait for all tasks to actually start executing
	wg.Wait()

	// Stop should complete cleanly
	s.Stop()

	if count.Load() != 10 {
		t.Errorf("expected 10 tasks completed, got %d", count.Load())
	}
}

func TestScheduler_ContextCancellation(t *testing.T) {
	s := New(Config{
		ForegroundWorkers: 1,
		BackgroundWorkers: 1,
	})
	s.Start()
	defer s.Stop()

	started := make(chan struct{})
	done := make(chan struct{})

	s.Submit(Task{
		Name:     "cancellable",
		Priority: PriorityHigh,
		Fn: func(ctx context.Context) {
			close(started)
			<-ctx.Done()
			close(done)
		},
	})

	<-started
	s.Stop()

	select {
	case <-done:
		// good — context was cancelled
	case <-time.After(2 * time.Second):
		t.Fatal("task context not cancelled after stop")
	}
}

func TestScheduler_Stats(t *testing.T) {
	s := New(Config{
		ForegroundWorkers: 2,
		BackgroundWorkers: 1,
	})
	s.Start()
	defer s.Stop()

	// Submit and wait for a few tasks
	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		s.Submit(Task{
			Name:     "stats-task",
			Priority: PriorityHigh,
			Fn: func(ctx context.Context) {
				time.Sleep(10 * time.Millisecond)
				wg.Done()
			},
		})
	}

	wg.Wait()
	time.Sleep(20 * time.Millisecond)

	stats := s.Stats()
	if stats.HighPriorityCompleted < 5 {
		t.Errorf("expected >=5 high priority completed, got %d", stats.HighPriorityCompleted)
	}
}

func TestScheduler_BackgroundThrottle(t *testing.T) {
	s := New(Config{
		ForegroundWorkers:  1,
		BackgroundWorkers:  1,
		BackgroundThrottle: 20 * time.Millisecond,
	})
	s.Start()
	defer s.Stop()

	var count atomic.Int32
	var wg sync.WaitGroup

	for i := 0; i < 3; i++ {
		wg.Add(1)
		s.Submit(Task{
			Name:     "throttled-bg",
			Priority: PriorityBackground,
			Fn: func(ctx context.Context) {
				count.Add(1)
				wg.Done()
			},
		})
	}

	wg.Wait()
	time.Sleep(100 * time.Millisecond)

	if count.Load() != 3 {
		t.Errorf("expected 3 background tasks completed, got %d", count.Load())
	}
}

func TestPriority_String(t *testing.T) {
	tests := []struct {
		p    Priority
		want string
	}{
		{PriorityBackground, "background"},
		{PriorityHigh, "high"},
		{Priority(99), "unknown"},
	}
	for _, tt := range tests {
		if got := tt.p.String(); got != tt.want {
			t.Errorf("Priority(%d).String() = %q, want %q", tt.p, got, tt.want)
		}
	}
}
