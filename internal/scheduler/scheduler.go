// Package scheduler provides a priority-aware task scheduler for background
// operations (compaction, GC, rebuild) that must not starve foreground
// request handling.
//
// Inspired by dbeel's glommio Shares/Latency system (src/args.rs:160-172),
// where background tasks get fewer CPU shares than foreground tasks.
//
// In Go, we approximate this with separate worker pools and configurable
// throttling for background tasks. Background tasks yield between work units
// via configurable sleep, preventing them from starving query goroutines.
package scheduler

import (
	"context"
	"log/slog"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// Priority determines which worker pool handles a task.
type Priority int

const (
	// PriorityBackground is for compaction, GC, and rebuild tasks.
	// These run with throttling to avoid starving request handlers.
	PriorityBackground Priority = iota

	// PriorityHigh is for request-serving tasks (HTTP handlers).
	// These run at full speed with no artificial delays.
	PriorityHigh
)

// String returns a human-readable name for the priority level.
func (p Priority) String() string {
	switch p {
	case PriorityBackground:
		return "background"
	case PriorityHigh:
		return "high"
	default:
		return "unknown"
	}
}

// Task is a unit of work submitted to the scheduler.
type Task struct {
	// Name is a human-readable identifier for logging.
	Name string

	// Priority determines which worker pool handles this task.
	Priority Priority

	// Fn is the work function. It receives a context that is cancelled
	// when the scheduler is stopped.
	Fn func(ctx context.Context)
}

// Stats reports scheduler activity.
type Stats struct {
	HighPriorityQueued    int
	BackgroundQueued      int
	HighPriorityCompleted int64
	BackgroundCompleted   int64
}

// Config controls the scheduler's behavior.
type Config struct {
	// ForegroundWorkers is the number of goroutines serving high-priority tasks.
	// Default: GOMAXPROCS.
	ForegroundWorkers int

	// BackgroundWorkers is the number of goroutines serving background tasks.
	// Default: max(1, GOMAXPROCS/4).
	BackgroundWorkers int

	// BackgroundThrottle is the sleep inserted between background task executions
	// to yield CPU to foreground tasks. Default: 10ms.
	BackgroundThrottle time.Duration
}

// Scheduler manages two worker pools with priority-aware dispatch.
type Scheduler struct {
	config    Config
	highPri   chan Task
	lowPri    chan Task
	ctx       context.Context
	cancel    context.CancelFunc
	wg        sync.WaitGroup
	completed [2]atomic.Int64 // [background, high]
}

// New creates a new Scheduler with the given config. Missing fields get defaults.
func New(cfg Config) *Scheduler {
	if cfg.ForegroundWorkers <= 0 {
		cfg.ForegroundWorkers = runtime.GOMAXPROCS(0)
	}
	if cfg.BackgroundWorkers <= 0 {
		cfg.BackgroundWorkers = max(1, runtime.GOMAXPROCS(0)/4)
	}
	if cfg.BackgroundThrottle <= 0 {
		cfg.BackgroundThrottle = 10 * time.Millisecond
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &Scheduler{
		config:  cfg,
		highPri: make(chan Task, 1024),
		lowPri:  make(chan Task, 1024),
		ctx:     ctx,
		cancel:  cancel,
	}
}

// Start launches all worker goroutines.
func (s *Scheduler) Start() {
	slog.Info("scheduler starting",
		"foreground_workers", s.config.ForegroundWorkers,
		"background_workers", s.config.BackgroundWorkers,
		"background_throttle", s.config.BackgroundThrottle,
	)

	// Foreground workers — no throttle, process high-priority tasks
	for i := 0; i < s.config.ForegroundWorkers; i++ {
		s.wg.Add(1)
		go s.foregroundWorker(i)
	}

	// Background workers — throttled, process low-priority tasks
	for i := 0; i < s.config.BackgroundWorkers; i++ {
		s.wg.Add(1)
		go s.backgroundWorker(i)
	}
}

// Stop cancels all workers and waits for in-flight tasks to finish.
func (s *Scheduler) Stop() {
	slog.Info("scheduler stopping")
	s.cancel()
	s.wg.Wait()
	slog.Info("scheduler stopped",
		"high_priority_completed", s.completed[PriorityHigh].Load(),
		"background_completed", s.completed[PriorityBackground].Load(),
	)
}

// Submit queues a task for execution. Non-blocking — the task is buffered.
func (s *Scheduler) Submit(t Task) {
	switch t.Priority {
	case PriorityHigh:
		s.highPri <- t
	case PriorityBackground:
		s.lowPri <- t
	default:
		slog.Warn("scheduler: unknown priority, treating as background",
			"task", t.Name,
			"priority", t.Priority,
		)
		s.lowPri <- t
	}
}

// Stats returns current scheduler statistics.
func (s *Scheduler) Stats() Stats {
	return Stats{
		HighPriorityQueued:    len(s.highPri),
		BackgroundQueued:      len(s.lowPri),
		HighPriorityCompleted: s.completed[PriorityHigh].Load(),
		BackgroundCompleted:   s.completed[PriorityBackground].Load(),
	}
}

// foregroundWorker processes high-priority tasks with no throttle.
func (s *Scheduler) foregroundWorker(id int) {
	defer s.wg.Done()
	for {
		select {
		case <-s.ctx.Done():
			return
		case task := <-s.highPri:
			s.runTask(task, PriorityHigh)
		}
	}
}

// backgroundWorker processes low-priority tasks with configurable throttle
// to yield CPU to foreground workers.
func (s *Scheduler) backgroundWorker(id int) {
	defer s.wg.Done()
	for {
		select {
		case <-s.ctx.Done():
			return
		case task := <-s.lowPri:
			s.runTask(task, PriorityBackground)
			// Throttle: yield CPU to foreground tasks
			select {
			case <-s.ctx.Done():
				return
			case <-time.After(s.config.BackgroundThrottle):
			}
		}
	}
}

// runTask executes a single task with panic recovery.
func (s *Scheduler) runTask(task Task, pri Priority) {
	defer func() {
		if r := recover(); r != nil {
			slog.Error("scheduler: task panicked",
				"task", task.Name,
				"priority", pri.String(),
				"recover", r,
			)
			buf := make([]byte, 4096)
			n := runtime.Stack(buf, false)
			slog.Error("scheduler: panic stack", "stack", string(buf[:n]))
		}
	}()

	task.Fn(s.ctx)
	s.completed[pri].Add(1)
}
