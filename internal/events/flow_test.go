package events

import (
	"context"
	"sync"
	"testing"
	"time"
)

func TestFlowBus_SubscribeAndNotify(t *testing.T) {
	bus := NewFlowBus()
	ch := bus.Subscribe(EventCollectionCreated)

	select {
	case <-ch:
		t.Fatal("received notification before notify")
	default:
		// good — not yet notified
	}

	bus.Notify(EventCollectionCreated)

	select {
	case <-ch:
		// good — notified
	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for notification")
	}
}

func TestFlowBus_NotifyBeforeSubscribe(t *testing.T) {
	bus := NewFlowBus()
	// Notify with no subscribers — should not panic
	bus.Notify(EventCollectionCreated)
}

func TestFlowBus_MultipleSubscribers(t *testing.T) {
	bus := NewFlowBus()
	ch1 := bus.Subscribe(EventCollectionCreated)
	ch2 := bus.Subscribe(EventCollectionCreated)

	bus.Notify(EventCollectionCreated)

	for i, ch := range []<-chan struct{}{ch1, ch2} {
		select {
		case <-ch:
			// good
		case <-time.After(1 * time.Second):
			t.Fatalf("subscriber %d timed out", i)
		}
	}
}

func TestFlowBus_NotifyDifferentEvent(t *testing.T) {
	bus := NewFlowBus()
	ch := bus.Subscribe(EventCollectionCreated)

	bus.Notify(EventCompactionComplete) // different event

	select {
	case <-ch:
		t.Fatal("received notification for wrong event")
	case <-time.After(50 * time.Millisecond):
		// good — not notified for different event
	}
}

func TestFlowBus_WaitForEvent(t *testing.T) {
	bus := NewFlowBus()

	go func() {
		time.Sleep(50 * time.Millisecond)
		bus.Notify(EventWALReplayComplete)
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	if err := bus.WaitForEvent(ctx, EventWALReplayComplete); err != nil {
		t.Fatalf("WaitForEvent failed: %v", err)
	}
}

func TestFlowBus_WaitForEvent_Timeout(t *testing.T) {
	bus := NewFlowBus()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	err := bus.WaitForEvent(ctx, EventCollectionCreated)
	if err == nil {
		t.Fatal("expected timeout error, got nil")
	}
}

func TestFlowBus_WaitForEvent_ContextCancelled(t *testing.T) {
	bus := NewFlowBus()

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	err := bus.WaitForEvent(ctx, EventCollectionCreated)
	if err == nil {
		t.Fatal("expected context cancellation error, got nil")
	}
}

func TestFlowBus_WaitForEvents(t *testing.T) {
	bus := NewFlowBus()

	go func() {
		time.Sleep(30 * time.Millisecond)
		bus.Notify(EventCollectionCreated)
		time.Sleep(30 * time.Millisecond)
		bus.Notify(EventVectorInserted)
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	if err := bus.WaitForEvents(ctx, EventCollectionCreated, EventVectorInserted); err != nil {
		t.Fatalf("WaitForEvents failed: %v", err)
	}
}

func TestFlowBus_OneShotSemantics(t *testing.T) {
	bus := NewFlowBus()
	ch1 := bus.Subscribe(EventCollectionCreated)

	bus.Notify(EventCollectionCreated)

	<-ch1

	// Second subscriber should NOT receive the old notification
	ch2 := bus.Subscribe(EventCollectionCreated)
	select {
	case <-ch2:
		t.Fatal("second subscriber received stale notification")
	case <-time.After(50 * time.Millisecond):
		// good — one-shot, old notification gone
	}
}

func TestFlowBus_ConcurrentNotify(t *testing.T) {
	bus := NewFlowBus()
	var wg sync.WaitGroup

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ch := bus.Subscribe(EventVectorInserted)
			bus.Notify(EventVectorInserted)
			<-ch
		}()
	}

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		// good
	case <-time.After(5 * time.Second):
		t.Fatal("concurrent test timed out")
	}
}

func TestFlowEvent_String(t *testing.T) {
	tests := []struct {
		event FlowEvent
		want  string
	}{
		{EventCollectionCreated, "collection_created"},
		{EventCompactionComplete, "compaction_complete"},
		{EventWALReplayComplete, "wal_replay_complete"},
		{EventVectorInserted, "vector_inserted"},
		{EventSegmentFlushed, "segment_flushed"},
		{EventGCComplete, "gc_complete"},
		{FlowEvent(999), "unknown"},
	}

	for _, tt := range tests {
		if got := tt.event.String(); got != tt.want {
			t.Errorf("FlowEvent(%d).String() = %q, want %q", tt.event, got, tt.want)
		}
	}
}
