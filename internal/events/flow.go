// Package events provides a lightweight publish-subscribe event bus for
// deterministic coordination between components and integration tests.
//
// Inspired by dbeel's FlowEvent system (src/flow_events.rs), which allows
// integration tests to wait for specific code paths without time.Sleep().
//
// Events are one-shot: a Notify() sends to all current subscribers and then
// clears the listener list. Subscribers that register after a Notify() do NOT
// receive the old notification.
package events

import (
	"context"
	"fmt"
	"sync"
)

// FlowEvent identifies a specific point in the system's execution that
// other components or tests can wait on.
type FlowEvent int

const (
	// EventCollectionCreated fires after a collection is created and
	// registered in the Manager's in-memory map.
	EventCollectionCreated FlowEvent = iota

	// EventCompactionComplete fires after a background compaction cycle
	// finishes successfully.
	EventCompactionComplete

	// EventWALReplayComplete fires after WAL replay finishes during startup.
	EventWALReplayComplete

	// EventVectorInserted fires after a batch of vectors is successfully
	// inserted into a collection (WAL + index + metadata).
	EventVectorInserted

	// EventSegmentFlushed fires after an in-memory index is flushed to
	// an immutable segment file on disk.
	EventSegmentFlushed

	// EventGCComplete fires after the garbage collector sweeps and
	// removes eligible segments.
	EventGCComplete
)

// String returns a human-readable name for the event.
func (e FlowEvent) String() string {
	switch e {
	case EventCollectionCreated:
		return "collection_created"
	case EventCompactionComplete:
		return "compaction_complete"
	case EventWALReplayComplete:
		return "wal_replay_complete"
	case EventVectorInserted:
		return "vector_inserted"
	case EventSegmentFlushed:
		return "segment_flushed"
	case EventGCComplete:
		return "gc_complete"
	default:
		return "unknown"
	}
}

// FlowBus is a thread-safe, one-shot publish-subscribe event bus.
//
// Usage:
//
//	bus := events.NewFlowBus()
//	ch := bus.Subscribe(events.EventCollectionCreated)
//	// ... trigger collection creation ...
//	<-ch // blocks until Notify(EventCollectionCreated) is called
type FlowBus struct {
	mu        sync.Mutex
	listeners map[FlowEvent][]chan struct{}
}

// NewFlowBus creates a new FlowBus.
func NewFlowBus() *FlowBus {
	return &FlowBus{
		listeners: make(map[FlowEvent][]chan struct{}),
	}
}

// Subscribe registers interest in an event and returns a channel that will
// receive exactly one value when Notify(event) is called. The channel is
// buffered with capacity 1 so Notify never blocks.
func (b *FlowBus) Subscribe(event FlowEvent) <-chan struct{} {
	b.mu.Lock()
	defer b.mu.Unlock()

	ch := make(chan struct{}, 1)
	b.listeners[event] = append(b.listeners[event], ch)
	return ch
}

// Notify sends a signal to all current subscribers of the given event
// and removes them from the listener list (one-shot semantics).
// Safe to call with no subscribers — a no-op in that case.
func (b *FlowBus) Notify(event FlowEvent) {
	b.mu.Lock()
	subs := b.listeners[event]
	delete(b.listeners, event)
	b.mu.Unlock()

	for _, ch := range subs {
		select {
		case ch <- struct{}{}:
		default:
			// Already signaled (shouldn't happen with one-shot, but be safe)
		}
	}
}

// WaitForEvent blocks until the given event is notified or the context is
// cancelled. Returns nil on success, ctx.Err() on cancellation/timeout.
func (b *FlowBus) WaitForEvent(ctx context.Context, event FlowEvent) error {
	ch := b.Subscribe(event)

	select {
	case <-ch:
		return nil
	case <-ctx.Done():
		return fmt.Errorf("waiting for %s: %w", event, ctx.Err())
	}
}

// WaitForEvents blocks until ALL given events are notified or the context is
// cancelled. Events may arrive in any order.
func (b *FlowBus) WaitForEvents(ctx context.Context, events ...FlowEvent) error {
	if len(events) == 0 {
		return nil
	}

	done := make(chan struct{})
	var mu sync.Mutex
	remaining := len(events)

	for _, event := range events {
		ch := b.Subscribe(event)
		go func() {
			select {
			case <-ch:
				mu.Lock()
				remaining--
				if remaining == 0 {
					close(done)
				}
				mu.Unlock()
			case <-ctx.Done():
				return
			}
		}()
	}

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return fmt.Errorf("waiting for %d events: %w", len(events), ctx.Err())
	}
}
