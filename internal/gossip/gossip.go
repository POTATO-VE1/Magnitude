// Package gossip implements a UDP-based gossip protocol for cluster membership
// dissemination, inspired by dbeel's gossip system (src/gossip.rs, src/tasks/gossip_server.rs).
//
// The gossip protocol enables decentralized cluster membership:
//   - Nodes broadcast Alive/Dead events when joining or leaving
//   - Events propagate through the cluster via configurable fanout
//   - Deduplication prevents infinite message loops
//   - Idempotent event handling tolerates duplicate delivery
//
// Wire format: JSON over UDP. Each message carries the source node ID,
// event type, optional payload, timestamp, and sequence number.
package gossip

import (
	"encoding/json"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"
)

// EventKind identifies the type of gossip event.
type EventKind uint8

const (
	// EventAlive signals that a node has joined the cluster.
	EventAlive EventKind = iota + 1

	// EventDead signals that a node has left or failed.
	EventDead

	// EventCreateCollection signals that a new collection was created.
	EventCreateCollection

	// EventDropCollection signals that a collection was dropped.
	EventDropCollection
)

// String returns a human-readable name for the event kind.
func (e EventKind) String() string {
	switch e {
	case EventAlive:
		return "alive"
	case EventDead:
		return "dead"
	case EventCreateCollection:
		return "create_collection"
	case EventDropCollection:
		return "drop_collection"
	default:
		return "unknown"
	}
}

// Message is a single gossip message sent over UDP.
type Message struct {
	Source    string    `json:"source"`    // node ID that originated the event
	Event     EventKind `json:"event"`     // event type
	Payload   []byte    `json:"payload"`   // optional event-specific data
	Timestamp time.Time `json:"timestamp"` // when the event occurred
	SeqNo     uint64    `json:"seq_no"`    // monotonic sequence number
}

// seenKey uniquely identifies a gossip event for deduplication.
// Includes SeqNo so that re-broadcasts (e.g., node restart sending a new
// Alive event) are not silently dropped as duplicates.
type seenKey struct {
	source string
	event  EventKind
	seqNo  uint64
}

// seenSet tracks which gossip events have been seen, with automatic expiry.
// Inspired by dbeel's gossip_requests HashMap with 30s expiration.
type seenSet struct {
	mu      sync.Mutex
	entries map[seenKey]time.Time
	maxSize int
	ttl     time.Duration
}

// newSeenSet creates a seen set with the given capacity and TTL.
func newSeenSet(maxSize int, ttl time.Duration) *seenSet {
	return &seenSet{
		entries: make(map[seenKey]time.Time, maxSize),
		maxSize: maxSize,
		ttl:     ttl,
	}
}

// hasSeen returns true if the key has been seen within the TTL window.
func (ss *seenSet) hasSeen(key seenKey) bool {
	ss.mu.Lock()
	defer ss.mu.Unlock()

	seenAt, exists := ss.entries[key]
	if !exists {
		return false
	}

	// Check TTL
	if time.Since(seenAt) > ss.ttl {
		delete(ss.entries, key)
		return false
	}

	return true
}

// markSeen records that the key was seen at the current time.
// If the set is at capacity, evicts the oldest entry.
func (ss *seenSet) markSeen(key seenKey) {
	ss.mu.Lock()
	defer ss.mu.Unlock()

	// Evict oldest if at capacity
	if len(ss.entries) >= ss.maxSize {
		ss.evictOldest()
	}

	ss.entries[key] = time.Now()
}

// evictOldest removes the oldest entry from the set.
// Must be called with ss.mu held.
func (ss *seenSet) evictOldest() {
	var oldestKey seenKey
	var oldestTime time.Time
	first := true

	for k, t := range ss.entries {
		if first || t.Before(oldestTime) {
			oldestKey = k
			oldestTime = t
			first = false
		}
	}

	if !first {
		delete(ss.entries, oldestKey)
	}
}

// size returns the number of entries in the set.
func (ss *seenSet) size() int {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	return len(ss.entries)
}

// EventCallback is called when a new (non-duplicate) gossip event is received.
type EventCallback func(msg Message)

// Config configures the gossip protocol.
type Config struct {
	// Port is the UDP port to listen on. Default: 7946
	Port int `yaml:"port"`

	// Fanout is the number of random peers to gossip to per round. Default: 3
	Fanout int `yaml:"fanout"`

	// MaxSeen is the maximum number of events to track for dedup. Default: 5000
	MaxSeen int `yaml:"maxSeen"`

	// SeenExpiry is the TTL for seen events. Default: 30s
	SeenExpiry time.Duration `yaml:"seenExpiry"`

	// ProbeInterval is how often to gossip. Default: 1s
	ProbeInterval time.Duration `yaml:"probeInterval"`
}

// DefaultConfig returns sensible defaults for the gossip protocol.
func DefaultConfig() Config {
	return Config{
		Port:          7946,
		Fanout:        3,
		MaxSeen:       5000,
		SeenExpiry:    30 * time.Second,
		ProbeInterval: 1 * time.Second,
	}
}

// Protocol manages the gossip protocol state.
type Protocol struct {
	config   Config
	nodeID   string
	seen     *seenSet
	seqNo    atomic.Uint64
	callback EventCallback
	peers    []string // known peer addresses (host:port)
	mu       sync.RWMutex
	running  bool
	stopCh   chan struct{}
	stopOnce sync.Once

	bufferMu sync.Mutex
	buffer   []Message
}

// New creates a new gossip protocol instance.
func New(nodeID string, cfg Config, callback EventCallback) *Protocol {
	if cfg.Fanout <= 0 {
		cfg.Fanout = 3
	}
	if cfg.MaxSeen <= 0 {
		cfg.MaxSeen = 5000
	}
	if cfg.SeenExpiry <= 0 {
		cfg.SeenExpiry = 30 * time.Second
	}
	if cfg.ProbeInterval <= 0 {
		cfg.ProbeInterval = 1 * time.Second
	}

	return &Protocol{
		config:   cfg,
		nodeID:   nodeID,
		seen:     newSeenSet(cfg.MaxSeen, cfg.SeenExpiry),
		callback: callback,
		stopCh:   make(chan struct{}),
		buffer:   make([]Message, 0, 100),
	}
}

// SetPeers updates the list of known peer addresses.
func (g *Protocol) SetPeers(peers []string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.peers = peers
}

// HandleMessage processes an incoming gossip message.
// Returns true if the message was new (not a duplicate), false if deduplicated.
func (g *Protocol) HandleMessage(msg Message) bool {
	key := seenKey{source: msg.Source, event: msg.Event, seqNo: msg.SeqNo}

	// Dedup check
	if g.seen.hasSeen(key) {
		return false
	}

	// Mark as seen
	g.seen.markSeen(key)

	// Don't process our own messages
	if msg.Source == g.nodeID {
		return true
	}

	// Deliver to callback (async to avoid blocking the UDP receive loop)
	if g.callback != nil {
		go g.callback(msg)
	}

	return true
}

// CreateMessage creates a new gossip message from this node.
func (g *Protocol) CreateMessage(event EventKind, payload []byte) Message {
	seq := g.seqNo.Add(1)
	return Message{
		Source:    g.nodeID,
		Event:     event,
		Payload:   payload,
		Timestamp: time.Now().Truncate(time.Second),
		SeqNo:     seq,
	}
}

// EncodeMessage serializes a message to JSON bytes for UDP transmission.
func EncodeMessage(msg Message) ([]byte, error) {
	return json.Marshal(msg)
}

// DecodeMessage deserializes a message from JSON bytes.
func DecodeMessage(data []byte) (Message, error) {
	var msg Message
	err := json.Unmarshal(data, &msg)
	return msg, err
}

// Stats returns gossip protocol statistics.
type Stats struct {
	SeenEntries int
	PeerCount   int
	SeqNo       uint64
	Running     bool
}

// GetStats returns current protocol statistics.
func (g *Protocol) GetStats() Stats {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return Stats{
		SeenEntries: g.seen.size(),
		PeerCount:   len(g.peers),
		SeqNo:       g.seqNo.Load(),
		Running:     g.running,
	}
}

// LogReceived logs a received gossip message (for observability).
func LogReceived(msg Message) {
	slog.Debug("gossip: received",
		"source", msg.Source,
		"event", msg.Event.String(),
		"seq_no", msg.SeqNo,
	)
}
