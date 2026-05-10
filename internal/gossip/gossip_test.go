package gossip

import (
	"encoding/json"
	"testing"
	"time"
)

func TestEventKind_String(t *testing.T) {
	tests := []struct {
		kind EventKind
		want string
	}{
		{EventAlive, "alive"},
		{EventDead, "dead"},
		{EventCreateCollection, "create_collection"},
		{EventDropCollection, "drop_collection"},
		{EventKind(99), "unknown"},
	}
	for _, tt := range tests {
		if got := tt.kind.String(); got != tt.want {
			t.Errorf("EventKind(%d).String() = %q, want %q", tt.kind, got, tt.want)
		}
	}
}

func TestMessage_Serialization(t *testing.T) {
	msg := Message{
		Source:    "node-1",
		Event:     EventAlive,
		Payload:   []byte(`{"id":"node-1","address":"10.0.0.1:7946"}`),
		Timestamp: time.Now().Truncate(time.Second),
		SeqNo:     42,
	}

	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	var decoded Message
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}

	if decoded.Source != msg.Source {
		t.Errorf("Source = %q, want %q", decoded.Source, msg.Source)
	}
	if decoded.Event != msg.Event {
		t.Errorf("Event = %d, want %d", decoded.Event, msg.Event)
	}
	if decoded.SeqNo != msg.SeqNo {
		t.Errorf("SeqNo = %d, want %d", decoded.SeqNo, msg.SeqNo)
	}
}

func TestSeenSet_Basic(t *testing.T) {
	ss := newSeenSet(100, 1*time.Hour)

	key := seenKey{source: "node-1", event: EventAlive}

	// Not seen initially
	if ss.hasSeen(key) {
		t.Error("expected not seen")
	}

	// Mark as seen
	ss.markSeen(key)

	// Now seen
	if !ss.hasSeen(key) {
		t.Error("expected seen")
	}
}

func TestSeenSet_Expiry(t *testing.T) {
	ss := newSeenSet(100, 50*time.Millisecond)

	key := seenKey{source: "node-1", event: EventAlive}
	ss.markSeen(key)

	if !ss.hasSeen(key) {
		t.Error("expected seen before expiry")
	}

	time.Sleep(60 * time.Millisecond)

	if ss.hasSeen(key) {
		t.Error("expected not seen after expiry")
	}
}

func TestSeenSet_MaxSize(t *testing.T) {
	ss := newSeenSet(3, 1*time.Hour)

	// Add 3 entries
	ss.markSeen(seenKey{source: "node-1", event: EventAlive})
	ss.markSeen(seenKey{source: "node-2", event: EventAlive})
	ss.markSeen(seenKey{source: "node-3", event: EventAlive})

	// All should be seen
	if !ss.hasSeen(seenKey{source: "node-1", event: EventAlive}) {
		t.Error("node-1 should be seen")
	}

	// Add a 4th — should evict oldest
	ss.markSeen(seenKey{source: "node-4", event: EventAlive})

	// Size should still be 3 (or less due to eviction)
	if ss.size() > 3 {
		t.Errorf("size = %d, want <= 3", ss.size())
	}
}

func TestSeenSet_Dedup(t *testing.T) {
	ss := newSeenSet(100, 1*time.Hour)

	key := seenKey{source: "node-1", event: EventAlive}

	// First time — not seen
	if ss.hasSeen(key) {
		t.Error("first check should be false")
	}

	// Mark seen
	ss.markSeen(key)

	// Second time — seen (dedup)
	if !ss.hasSeen(key) {
		t.Error("second check should be true")
	}

	// Third time — still seen
	if !ss.hasSeen(key) {
		t.Error("third check should be true")
	}
}

func TestSeenKey_DifferentEvents(t *testing.T) {
	ss := newSeenSet(100, 1*time.Hour)

	ss.markSeen(seenKey{source: "node-1", event: EventAlive})

	// Different event from same source — not seen
	if ss.hasSeen(seenKey{source: "node-1", event: EventDead}) {
		t.Error("different event should not be seen")
	}

	// Same event from different source — not seen
	if ss.hasSeen(seenKey{source: "node-2", event: EventAlive}) {
		t.Error("different source should not be seen")
	}
}
