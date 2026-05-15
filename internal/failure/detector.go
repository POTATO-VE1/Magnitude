// Package failure implements a failure detection system for cluster nodes,
// inspired by dbeel's failure_detector (src/tasks/failure_detector.rs).
//
// The detector uses a heartbeat-based approach with three states:
//   - Alive: node is responding to probes
//   - Suspect: node missed recent heartbeats (may be slow or network issue)
//   - Dead: node hasn't responded for too long (removed from routing)
//
// The detector runs a background goroutine that periodically checks node
// timestamps and transitions states. Callbacks notify the coordinator
// when nodes change state.
package failure

import (
	"log/slog"
	"math"
	"sync"
	"time"
)

// NodeState represents the health state of a cluster node.
type NodeState int

const (
	// StateUnknown means the node is not tracked by the detector.
	StateUnknown NodeState = iota

	// StateAlive means the node is healthy and responding.
	StateAlive

	// StateSuspect means the node missed recent heartbeats.
	StateSuspect

	// StateDead means the node has been unresponsive for too long.
	StateDead
)

// String returns a human-readable name for the node state.
func (s NodeState) String() string {
	switch s {
	case StateAlive:
		return "alive"
	case StateSuspect:
		return "suspect"
	case StateDead:
		return "dead"
	default:
		return "unknown"
	}
}

// trackedNode holds the state for a single monitored node.
type trackedNode struct {
	address     string
	state       NodeState
	lastSeen    time.Time
	stateChange time.Time

	// Phi-Accrual state
	intervals   []time.Duration
	intervalIdx int
	phi         float64
}

// Config configures the failure detector.
type Config struct {
	// Interval is how often to check node health. Default: 1s
	Interval time.Duration

	// Timeout is the probe timeout for individual nodes. Default: 500ms
	Timeout time.Duration

	// SuspectAfter is how long since last heartbeat before marking suspect. Default: 5s
	SuspectAfter time.Duration

	// DeadAfter is how long since last heartbeat before marking dead. Default: 15s
	DeadAfter time.Duration

	// OnNodeSuspect is called when a node transitions to Suspect state.
	OnNodeSuspect func(nodeID string)

	// OnNodeDead is called when a node transitions to Dead state.
	OnNodeDead func(nodeID string)
}

// DefaultConfig returns sensible defaults for the failure detector.
func DefaultConfig() Config {
	return Config{
		Interval:     1 * time.Second,
		Timeout:      500 * time.Millisecond,
		SuspectAfter: 5 * time.Second,
		DeadAfter:    15 * time.Second,
	}
}

// Detector monitors cluster node health and triggers state transitions.
type Detector struct {
	mu      sync.RWMutex
	config  Config
	nodes   map[string]*trackedNode
	running bool
	stopCh  chan struct{}
}

// New creates a new failure detector with the given config.
func New(cfg Config) *Detector {
	if cfg.Interval <= 0 {
		cfg.Interval = 1 * time.Second
	}
	if cfg.SuspectAfter <= 0 {
		cfg.SuspectAfter = 5 * time.Second
	}
	if cfg.DeadAfter <= 0 {
		cfg.DeadAfter = 15 * time.Second
	}

	return &Detector{
		config: cfg,
		nodes:  make(map[string]*trackedNode),
		stopCh: make(chan struct{}),
	}
}

// Start begins the background health checking loop.
func (d *Detector) Start() {
	d.mu.Lock()
	if d.running {
		d.mu.Unlock()
		return
	}
	d.running = true
	d.mu.Unlock()

	go d.checkLoop()
	slog.Info("failure detector started",
		"interval", d.config.Interval,
		"suspect_after", d.config.SuspectAfter,
		"dead_after", d.config.DeadAfter,
	)
}

// Stop halts the background health checking loop.
func (d *Detector) Stop() {
	d.mu.Lock()
	defer d.mu.Unlock()

	if !d.running {
		return
	}
	d.running = false
	close(d.stopCh)
	slog.Info("failure detector stopped")
}

// AddNode registers a node for monitoring.
func (d *Detector) AddNode(nodeID, address string) {
	d.mu.Lock()
	defer d.mu.Unlock()

	d.nodes[nodeID] = &trackedNode{
		address:  address,
		state:    StateAlive,
		lastSeen: time.Now(),
	}
	slog.Debug("failure detector: added node", "id", nodeID, "address", address)
}

// RemoveNode stops monitoring a node.
func (d *Detector) RemoveNode(nodeID string) {
	d.mu.Lock()
	defer d.mu.Unlock()

	delete(d.nodes, nodeID)
	slog.Debug("failure detector: removed node", "id", nodeID)
}

// RecordHeartbeat records that a node is alive (called when we receive
// any message from the node).
func (d *Detector) RecordHeartbeat(nodeID string) {
	d.mu.Lock()
	defer d.mu.Unlock()

	node, exists := d.nodes[nodeID]
	if !exists {
		return
	}

	now := time.Now()
	elapsed := now.Sub(node.lastSeen)
	node.lastSeen = now

	if node.intervals == nil {
		node.intervals = make([]time.Duration, 100)
	}
	// Ignore giant leaps (e.g. sleep/wake or first heartbeat)
	if elapsed > 0 && elapsed < 5*time.Minute {
		node.intervals[node.intervalIdx] = elapsed
		node.intervalIdx = (node.intervalIdx + 1) % 100
	}

	if node.state != StateAlive {
		slog.Info("failure detector: node recovered",
			"id", nodeID,
			"previous_state", node.state.String(),
		)
		node.state = StateAlive
		node.stateChange = now
	}
}

// GetState returns the current state of a node.
func (d *Detector) GetState(nodeID string) NodeState {
	d.mu.RLock()
	defer d.mu.RUnlock()

	node, exists := d.nodes[nodeID]
	if !exists {
		return StateUnknown
	}
	return node.state
}

// GetAddress returns the HTTP API address of a node.
func (d *Detector) GetAddress(nodeID string) string {
	d.mu.RLock()
	defer d.mu.RUnlock()

	node, exists := d.nodes[nodeID]
	if !exists {
		return ""
	}
	return node.address
}

// NodeCount returns the number of tracked nodes.
func (d *Detector) NodeCount() int {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return len(d.nodes)
}

// GetAliveNodes returns the IDs of all nodes in Alive state.
func (d *Detector) GetAliveNodes() []string {
	d.mu.RLock()
	defer d.mu.RUnlock()

	var result []string
	for id, node := range d.nodes {
		if node.state == StateAlive {
			result = append(result, id)
		}
	}
	return result
}

// checkLoop periodically evaluates node health and triggers state transitions.
func (d *Detector) checkLoop() {
	ticker := time.NewTicker(d.config.Interval)
	defer ticker.Stop()

	for {
		select {
		case <-d.stopCh:
			return
		case <-ticker.C:
			d.checkAll()
		}
	}
}

// checkAll evaluates all tracked nodes and updates their state.
func (d *Detector) checkAll() {
	d.mu.Lock()
	defer d.mu.Unlock()

	now := time.Now()
	for nodeID, node := range d.nodes {
		elapsed := now.Sub(node.lastSeen)
		previousState := node.state

		var sum time.Duration
		var count int
		for _, inv := range node.intervals {
			if inv > 0 {
				sum += inv
				count++
			}
		}

		if count < 10 {
			// Not enough data for Phi calculation; fallback to static timeouts
			switch {
			case elapsed >= d.config.DeadAfter:
				node.state = StateDead
			case elapsed >= d.config.SuspectAfter:
				node.state = StateSuspect
			default:
				node.state = StateAlive
			}
		} else {
			// Phi-Accrual using Exponential Distribution
			mean := float64(sum) / float64(count)
			phi := (float64(elapsed) / mean) / math.Ln10
			node.phi = phi

			switch {
			case phi >= 8.0: // 99.999999% probability of failure
				node.state = StateDead
			case phi >= 3.0: // 99.9% probability
				node.state = StateSuspect
			default:
				node.state = StateAlive
			}
		}

		// Fire callbacks on state transitions
		if node.state != previousState {
			node.stateChange = now
			slog.Info("failure detector: state change",
				"node", nodeID,
				"from", previousState.String(),
				"to", node.state.String(),
				"elapsed", elapsed,
				"phi", node.phi,
			)

			switch node.state {
			case StateSuspect:
				if d.config.OnNodeSuspect != nil {
					go d.config.OnNodeSuspect(nodeID)
				}
			case StateDead:
				if d.config.OnNodeDead != nil {
					go d.config.OnNodeDead(nodeID)
				}
			}
		}
	}
}
