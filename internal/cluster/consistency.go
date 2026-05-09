// Package cluster — consistency levels for distributed operations.
//
// Inspired by dbeel's Consistency enum (dbeel_client/src/lib.rs:465-480):
//   - ConsistencyOne: single replica (fastest, no durability guarantee)
//   - ConsistencyQuorum: majority ack (balanced)
//   - ConsistencyAll: all replicas ack (slowest, strongest guarantee)
//
// In single-node mode, all levels behave identically (local operation only).
// In distributed mode, the level determines how many replicas must respond
// before the operation is acknowledged.
package cluster

import "fmt"

// ConsistencyLevel defines how many replicas must acknowledge an operation.
type ConsistencyLevel int

const (
	// ConsistencyOne requires exactly one replica to ack (the local node).
	// Fastest option — no inter-node coordination.
	ConsistencyOne ConsistencyLevel = iota

	// ConsistencyQuorum requires a majority (RF/2 + 1) of replicas to ack.
	// Balanced tradeoff between latency and durability.
	ConsistencyQuorum

	// ConsistencyAll requires all replicas to ack.
	// Strongest guarantee but highest latency.
	ConsistencyAll
)

// String returns the human-readable name for the consistency level.
func (c ConsistencyLevel) String() string {
	switch c {
	case ConsistencyOne:
		return "one"
	case ConsistencyQuorum:
		return "quorum"
	case ConsistencyAll:
		return "all"
	default:
		return "unknown"
	}
}

// ParseConsistencyLevel parses a string into a ConsistencyLevel.
// Empty string defaults to ConsistencyOne.
func ParseConsistencyLevel(s string) (ConsistencyLevel, error) {
	switch s {
	case "one", "":
		return ConsistencyOne, nil
	case "quorum":
		return ConsistencyQuorum, nil
	case "all":
		return ConsistencyAll, nil
	default:
		return 0, fmt.Errorf("invalid consistency level %q, must be one of [one, quorum, all]", s)
	}
}

// ResolveConsistency converts a ConsistencyLevel into an actual node count
// given the replication factor. The result is clamped to [1, rf].
func ResolveConsistency(level ConsistencyLevel, rf int) int {
	if rf <= 0 {
		rf = 1
	}
	switch level {
	case ConsistencyQuorum:
		q := rf/2 + 1
		if q > rf {
			q = rf
		}
		return q
	case ConsistencyAll:
		return rf
	default: // ConsistencyOne
		return 1
	}
}
