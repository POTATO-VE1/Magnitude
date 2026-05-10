package failure

import (
	"testing"
	"time"
)

func TestNodeState_String(t *testing.T) {
	tests := []struct {
		state NodeState
		want  string
	}{
		{StateAlive, "alive"},
		{StateSuspect, "suspect"},
		{StateDead, "dead"},
		{NodeState(99), "unknown"},
	}
	for _, tt := range tests {
		if got := tt.state.String(); got != tt.want {
			t.Errorf("NodeState(%d).String() = %q, want %q", tt.state, got, tt.want)
		}
	}
}

func TestDetector_Basic(t *testing.T) {
	d := New(Config{
		Interval:     100 * time.Millisecond,
		Timeout:      50 * time.Millisecond,
		SuspectAfter: 200 * time.Millisecond,
		DeadAfter:    500 * time.Millisecond,
	})

	// Add a node
	d.AddNode("node-1", "10.0.0.1:7946")

	// Should be alive initially
	state := d.GetState("node-1")
	if state != StateAlive {
		t.Errorf("initial state = %v, want alive", state)
	}

	d.Stop()
}

func TestDetector_RecordHeartbeat(t *testing.T) {
	d := New(Config{
		Interval:     100 * time.Millisecond,
		Timeout:      50 * time.Millisecond,
		SuspectAfter: 200 * time.Millisecond,
		DeadAfter:    500 * time.Millisecond,
	})
	defer d.Stop()

	d.AddNode("node-1", "10.0.0.1:7946")

	// Record heartbeat
	d.RecordHeartbeat("node-1")

	state := d.GetState("node-1")
	if state != StateAlive {
		t.Errorf("after heartbeat: state = %v, want alive", state)
	}
}

func TestDetector_SuspectTimeout(t *testing.T) {
	d := New(Config{
		Interval:     50 * time.Millisecond,
		Timeout:      25 * time.Millisecond,
		SuspectAfter: 100 * time.Millisecond,
		DeadAfter:    500 * time.Millisecond,
	})
	defer d.Stop()

	d.AddNode("node-1", "10.0.0.1:7946")
	d.Start()

	// Wait for suspect timeout
	time.Sleep(200 * time.Millisecond)

	state := d.GetState("node-1")
	if state != StateSuspect {
		t.Errorf("after timeout: state = %v, want suspect", state)
	}
}

func TestDetector_HeartbeatResetsState(t *testing.T) {
	d := New(Config{
		Interval:     50 * time.Millisecond,
		Timeout:      25 * time.Millisecond,
		SuspectAfter: 100 * time.Millisecond,
		DeadAfter:    500 * time.Millisecond,
	})
	defer d.Stop()

	d.AddNode("node-1", "10.0.0.1:7946")
	d.Start()

	// Wait for suspect
	time.Sleep(200 * time.Millisecond)
	if d.GetState("node-1") != StateSuspect {
		t.Fatal("expected suspect")
	}

	// Heartbeat should reset to alive
	d.RecordHeartbeat("node-1")
	if d.GetState("node-1") != StateAlive {
		t.Errorf("after heartbeat: state = %v, want alive", d.GetState("node-1"))
	}
}

func TestDetector_RemoveNode(t *testing.T) {
	d := New(Config{
		Interval:     100 * time.Millisecond,
		Timeout:      50 * time.Millisecond,
		SuspectAfter: 200 * time.Millisecond,
		DeadAfter:    500 * time.Millisecond,
	})
	defer d.Stop()

	d.AddNode("node-1", "10.0.0.1:7946")
	d.RemoveNode("node-1")

	// Should return unknown for removed node
	state := d.GetState("node-1")
	if state != StateUnknown {
		t.Errorf("removed node state = %v, want unknown", state)
	}
}

func TestDetector_GetUnknown(t *testing.T) {
	d := New(Config{
		Interval:     100 * time.Millisecond,
		Timeout:      50 * time.Millisecond,
		SuspectAfter: 200 * time.Millisecond,
		DeadAfter:    500 * time.Millisecond,
	})
	defer d.Stop()

	state := d.GetState("nonexistent")
	if state != StateUnknown {
		t.Errorf("unknown node state = %v, want unknown", state)
	}
}

func TestDetector_DeadCallback(t *testing.T) {
	deadCh := make(chan string, 10)

	d := New(Config{
		Interval:     50 * time.Millisecond,
		Timeout:      25 * time.Millisecond,
		SuspectAfter: 100 * time.Millisecond,
		DeadAfter:    200 * time.Millisecond,
		OnNodeDead: func(nodeID string) {
			deadCh <- nodeID
		},
	})
	defer d.Stop()

	d.AddNode("node-1", "10.0.0.1:7946")
	d.Start()

	// Wait for dead timeout
	select {
	case nodeID := <-deadCh:
		if nodeID != "node-1" {
			t.Errorf("dead callback: got %q, want node-1", nodeID)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("dead callback not called within timeout")
	}
}

func TestDetector_MultipleNodes(t *testing.T) {
	d := New(Config{
		Interval:     100 * time.Millisecond,
		Timeout:      50 * time.Millisecond,
		SuspectAfter: 200 * time.Millisecond,
		DeadAfter:    500 * time.Millisecond,
	})
	defer d.Stop()

	d.AddNode("node-1", "10.0.0.1:7946")
	d.AddNode("node-2", "10.0.0.2:7946")
	d.AddNode("node-3", "10.0.0.3:7946")

	// All should be alive
	for _, id := range []string{"node-1", "node-2", "node-3"} {
		if d.GetState(id) != StateAlive {
			t.Errorf("%s: expected alive", id)
		}
	}

	// Heartbeat only node-1 and node-2
	d.RecordHeartbeat("node-1")
	d.RecordHeartbeat("node-2")

	// Remove node-3
	d.RemoveNode("node-3")

	if d.NodeCount() != 2 {
		t.Errorf("node count = %d, want 2", d.NodeCount())
	}
}

func TestDetector_SuspectCallback(t *testing.T) {
	suspectCh := make(chan string, 10)

	d := New(Config{
		Interval:     50 * time.Millisecond,
		Timeout:      25 * time.Millisecond,
		SuspectAfter: 100 * time.Millisecond,
		DeadAfter:    500 * time.Millisecond,
		OnNodeSuspect: func(nodeID string) {
			suspectCh <- nodeID
		},
	})
	defer d.Stop()

	d.AddNode("node-1", "10.0.0.1:7946")
	d.Start()

	select {
	case nodeID := <-suspectCh:
		if nodeID != "node-1" {
			t.Errorf("suspect callback: got %q, want node-1", nodeID)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("suspect callback not called within timeout")
	}
}
