package gossip

import (
	"encoding/json"
	"sync"
	"sync/atomic"
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

// ── Regression Tests for BUG 1: SeqNo in dedup key ──────────────────────────

func TestHandleMessage_DifferentSeqNo_BothProcessed(t *testing.T) {
	// BUG 1 regression: same source + same event type but different SeqNo
	// must both be processed (not deduplicated).
	var callCount atomic.Int64
	var wg sync.WaitGroup
	wg.Add(2)
	p := New("self", DefaultConfig(), func(msg Message) {
		callCount.Add(1)
		wg.Done()
	})

	msg1 := Message{Source: "node-1", Event: EventAlive, SeqNo: 1}
	msg2 := Message{Source: "node-1", Event: EventAlive, SeqNo: 2}

	if !p.HandleMessage(msg1) {
		t.Error("msg1 should be new")
	}
	if !p.HandleMessage(msg2) {
		t.Error("msg2 should be new (different SeqNo)")
	}
	wg.Wait()
	if callCount.Load() != 2 {
		t.Errorf("callback called %d times, want 2", callCount.Load())
	}
}

func TestHandleMessage_SameSeqNo_Deduplicated(t *testing.T) {
	var callCount atomic.Int64
	var wg sync.WaitGroup
	wg.Add(1)
	p := New("self", DefaultConfig(), func(msg Message) {
		callCount.Add(1)
		wg.Done()
	})

	msg := Message{Source: "node-1", Event: EventAlive, SeqNo: 42}

	if !p.HandleMessage(msg) {
		t.Error("first HandleMessage should return true")
	}
	wg.Wait()
	if p.HandleMessage(msg) {
		t.Error("second HandleMessage with same SeqNo should return false")
	}
	// Give a small window for any erroneous async callback
	time.Sleep(10 * time.Millisecond)
	if callCount.Load() != 1 {
		t.Errorf("callback called %d times, want 1", callCount.Load())
	}
}

// ── Regression Tests for BUG 2: Buffer draining ─────────────────────────────

func TestBroadcast_BufferDrains(t *testing.T) {
	// BUG 2 regression: after dissemination, buffer should be empty.
	p := New("self", DefaultConfig(), nil)

	p.Broadcast(EventAlive, []byte("test"))

	// Buffer should have 1 message
	p.bufferMu.Lock()
	count := len(p.buffer)
	p.bufferMu.Unlock()
	if count != 1 {
		t.Errorf("buffer size = %d after Broadcast, want 1", count)
	}

	// Simulate dissemination (no peers, so nothing actually sent, but buffer drains)
	p.disseminate(nil, "")

	p.bufferMu.Lock()
	count = len(p.buffer)
	p.bufferMu.Unlock()
	if count != 0 {
		t.Errorf("buffer size = %d after disseminate, want 0 (should be drained)", count)
	}
}

// ── Regression Tests for BUG 3: Double-stop safety ──────────────────────────

func TestStopUDP_DoubleStop_NoPanic(t *testing.T) {
	// BUG 3 regression: calling StopUDP twice must not panic.
	p := New("self", DefaultConfig(), nil)

	// Don't actually start UDP — just test the stop path
	p.StopUDP()
	p.StopUDP() // second call must not panic
}

// ── HMAC Tests ──────────────────────────────────────────────────────────────

func TestSignVerify_Roundtrip(t *testing.T) {
	msg := &Message{
		Source: "node-1",
		Event:  EventAlive,
		SeqNo:  1,
	}
	secret := "test-secret-key"

	signed, err := SignMessage(msg, secret)
	if err != nil {
		t.Fatalf("SignMessage: %v", err)
	}

	decoded, err := VerifyAndUnmarshal(signed, secret)
	if err != nil {
		t.Fatalf("VerifyAndUnmarshal: %v", err)
	}
	if decoded.Source != "node-1" {
		t.Errorf("Source = %q, want node-1", decoded.Source)
	}
}

func TestSignVerify_TamperedMessage(t *testing.T) {
	msg := &Message{
		Source: "node-1",
		Event:  EventAlive,
		SeqNo:  1,
	}
	secret := "test-secret-key"

	signed, err := SignMessage(msg, secret)
	if err != nil {
		t.Fatalf("SignMessage: %v", err)
	}

	// Tamper with the payload (flip a byte before the HMAC suffix)
	if len(signed) > 32 {
		signed[0] ^= 0xFF
	}

	_, err = VerifyAndUnmarshal(signed, secret)
	if err == nil {
		t.Fatal("expected verification failure for tampered message")
	}
}

func TestSignVerify_NoSecret(t *testing.T) {
	msg := &Message{
		Source: "node-1",
		Event:  EventAlive,
		SeqNo:  1,
	}

	signed, err := SignMessage(msg, "")
	if err != nil {
		t.Fatalf("SignMessage: %v", err)
	}

	decoded, err := VerifyAndUnmarshal(signed, "")
	if err != nil {
		t.Fatalf("VerifyAndUnmarshal: %v", err)
	}
	if decoded.Source != "node-1" {
		t.Errorf("Source = %q, want node-1", decoded.Source)
	}
}
