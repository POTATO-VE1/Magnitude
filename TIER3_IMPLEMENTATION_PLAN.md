# Tier 3 Implementation Plan: Gossip, Failure Detection, Shard-Level Routing

## Executive Summary

Port three interrelated distributed systems features from dbeel (Rust/glommio) to Magnitude VectorDB (Go/net-http). These features transform Magnitude from a single-node database into a distributed cluster with automatic failure recovery and data migration.

**dbeel Architecture (Reference)**:
- Thread-per-core with glommio async runtime (single-threaded executors per CPU)
- `Rc<MyShard>` holds all state — no `Arc`, no cross-thread mutexes
- Gossip over UDP, shard RPC over TCP with bincode serialization
- Murmur3 consistent hashing for key→shard routing
- LSM-tree storage engine with per-collection trees

**Magnitude Architecture (Target)**:
- Standard Go goroutine-based concurrency with `sync.RWMutex`
- SQLite-backed SysDB and WAL
- In-memory HNSW/IVF/Flat indexes
- HTTP/JSON API via chi router
- No existing inter-node communication

---

## Feature 1: Gossip Protocol for Metadata Dissemination

### A. Current State Analysis

**dbeel gossip.rs** (`/home/veda/dbeel/src/gossip.rs:1-40`):
```
GossipEvent enum:
  - Alive(NodeMetadata)    — node join announcement
  - Dead(String)           — node death declaration  
  - CreateCollection(String, u16) — collection creation (name, replication_factor)
  - DropCollection(String) — collection deletion

GossipMessage struct:
  - source: String         — originating node name
  - event: GossipEvent     — the event payload

Serialization: bincode (binary, compact)
```

**dbeel gossip_server.rs** (`/home/veda/dbeel/src/tasks/gossip_server.rs:1-112`):
- UDP server on `gossip_port` (default 30000)
- 65536-byte packet buffer
- Dedup via `gossip_requests: HashMap<(source, event_kind), seen_count>`
- `gossip_max_seen_count` threshold (default 3) before stopping retransmission
- 30-second expiry for dedup entries (spawned as async task)
- On first receipt: broadcast to local shards, handle event, forward to fanout peers
- On subsequent receipts: only forward if not yet at max seen count

**dbeel shards.rs gossip methods** (`/home/veda/dbeel/src/shards.rs:791-827`):
- `gossip(event)`: serialize → broadcast to local shards → send to random fanout peers
- `gossip_buffer(msg)`: select `gossip_fanout` random nodes → send UDP packets in parallel
- Fanout selection: `rand::seq::IteratorRandom::choose_multiple` on node list

**Magnitude cluster/coordinator.go** (`/home/veda/Desktop/vector_db/internal/cluster/coordinator.go:1-276`):
- Already has `NodeState` enum: `NodeAlive`, `NodeSuspect`, `NodeDead`
- Already has `NodeInfo` struct with ID, Address, State, LastHeartbeat
- Already has `Coordinator` with `JoinNode`, `LeaveNode`, `Heartbeat`, `CheckHealth`
- Already has `HashRing` with consistent hashing
- **Missing**: No gossip protocol, no UDP server, no event dissemination

### B. Architecture Design

**New package**: `internal/gossip/`

```
internal/gossip/
├── event.go        — GossipEvent types and serialization
├── server.go       — UDP gossip server
├── disseminator.go — Fanout selection and send logic
├── dedup.go        — Seen-count deduplication with expiry
└── config.go       — Gossip configuration
```

**New interfaces and types**:

```go
// event.go
type GossipEventType uint8

const (
    GossipAlive GossipEventType = iota + 1
    GossipDead
    GossipCreateCollection
    GossipDropCollection
)

type GossipEvent struct {
    Type             GossipEventType
    NodeMetadata     *NodeMetadata     // for Alive
    NodeName         string            // for Dead
    CollectionName   string            // for Create/Drop
    ReplicationFactor uint16           // for Create
}

type GossipMessage struct {
    Source  string       // originating node name
    Event   GossipEvent
    SeqNo   uint64       // monotonic sequence for dedup
}

type NodeMetadata struct {
    Name        string   `json:"name"`
    Address     string   `json:"address"`      // host:port for RPC
    APIAddress  string   `json:"api_address"`   // host:port for client API
    GossipPort  int      `json:"gossip_port"`
}
```

**Communication pattern**:
- UDP for gossip (fire-and-forget, low overhead)
- JSON serialization (not bincode — Go ecosystem compatibility)
- Channels for intra-process event delivery to collection manager

**Goroutines**:
1. `gossipServer` — UDP listener, receives and processes gossip packets
2. `gossipDisseminator` — periodic fanout retransmission of buffered events
3. `gossipDedupCleaner` — periodic cleanup of expired dedup entries

### C. Implementation Steps

#### Phase 1: Foundation (Interfaces, Types, Config)

**Step 1.1**: Create `internal/gossip/event.go`
- Define `GossipEventType`, `GossipEvent`, `GossipMessage`, `NodeMetadata` types
- Implement `Encode() ([]byte, error)` and `Decode([]byte) (*GossipMessage, error)` using `encoding/json`
- Add monotonic `SeqNo` field for dedup

**Step 1.2**: Create `internal/gossip/config.go`
```go
type Config struct {
    Port              int           `yaml:"port"`              // UDP listen port (default: 30000)
    Fanout            int           `yaml:"fanout"`            // peers per gossip round (default: 3)
    MaxSeenCount      int           `yaml:"maxSeenCount"`      // dedup threshold (default: 3)
    SeenExpiry        time.Duration `yaml:"seenExpiry"`        // dedup entry TTL (default: 30s)
    RetransmitInterval time.Duration `yaml:"retransmitInterval"` // gossip interval (default: 500ms)
}
```

**Step 1.3**: Add `Gossip` section to `internal/config/config.go`
- Add `Gossip GossipConfig` field to `Config` struct
- Add defaults in `DefaultConfig()`
- Add validation in `Validate()`

#### Phase 2: Core Logic

**Step 2.1**: Create `internal/gossip/dedup.go`
- `DedupTracker` struct with `map[string]uint8` (key: "source:eventType:seqNo")
- `ShouldProcess(msg) bool` — returns true if seen count < maxSeenCount
- `Cleanup()` — remove entries older than SeenExpiry
- Thread-safe with `sync.RWMutex`

**Step 2.2**: Create `internal/gossip/disseminator.go`
- `Disseminator` struct holding node list reference, fanout config, UDP socket pool
- `SelectPeers(nodes []NodeMetadata, fanout int) []NodeMetadata` — random selection without self
- `Broadcast(msg GossipMessage) error` — encode + send to fanout peers via UDP
- `BroadcastToAll(msg GossipMessage) error` — send to all peers (for critical events like Dead)

**Step 2.3**: Create `internal/gossip/server.go`
- `Server` struct with UDP conn, dedup tracker, event handler callback, disseminator
- `Start(ctx context.Context) error` — bind UDP socket, start receive loop
- `handlePacket(data []byte) error` — decode → dedup check → call handler → disseminate
- `Stop() error` — close UDP conn, cancel context

**Step 2.4**: Create `internal/gossip/event_handler.go`
- `EventHandler` interface:
  ```go
  type EventHandler interface {
      HandleAlive(node NodeMetadata) error
      HandleDead(nodeName string) error
      HandleCreateCollection(name string, replicationFactor uint16) error
      HandleDropCollection(name string) error
  }
  ```
- Implementations wired to Coordinator and Collection Manager

#### Phase 3: Integration

**Step 3.1**: Wire gossip into `cmd/server/main.go`
- After Coordinator creation, create `gossip.Server` with event handler
- Start gossip server goroutine
- On startup: broadcast `GossipAlive` with this node's metadata
- On shutdown: broadcast `GossipDead`

**Step 3.2**: Add inter-node RPC client in `internal/cluster/client.go`
- TCP-based request/response for shard operations (Ping, GetMetadata, GetCollections)
- Used by failure detector and migration engine
- Message framing: 4-byte length prefix + JSON payload

**Step 3.3**: Add inter-node RPC server in `internal/cluster/rpc_server.go`
- TCP listener for inter-node requests
- Routes to collection manager for shard operations
- Started alongside gossip server

**Step 3.4**: Modify `internal/collection/collection.go`
- Add `CreateCollectionRemote(name, dim, metric, indexType string)` — creates collection without gossip (called by gossip handler)
- Add `DeleteCollectionRemote(id string)` — deletes without gossip

### D. Gossip Protocol Design

**Message Format (JSON over UDP)**:
```json
{
    "source": "node-1",
    "seq_no": 42,
    "event": {
        "type": "alive",
        "node": {
            "name": "node-1",
            "address": "10.0.0.1:20000",
            "api_address": "10.0.0.1:8443",
            "gossip_port": 30000
        }
    }
}
```

**Event Types**:
| Type | Payload | When Sent | Handling |
|------|---------|-----------|----------|
| `alive` | NodeMetadata | Node startup, self-healing | Add to ring, trigger migration |
| `dead` | NodeName | Failure detector fires | Remove from ring, trigger migration |
| `create_collection` | name, replication_factor | Collection created | Create locally if not exists |
| `drop_collection` | name | Collection deleted | Delete locally if exists |

**Dedup Mechanism**:
- Key: `{source}:{event_type}:{seq_no}`
- Value: `seen_count` (uint8)
- Threshold: 3 (configurable via `gossip_max_seen_count`)
- Expiry: 30 seconds after last seen
- On threshold reached: schedule cleanup goroutine, stop retransmitting

**Fanout Selection Algorithm**:
```
func SelectPeers(allNodes []NodeMetadata, selfName string, fanout int) []NodeMetadata {
    // Filter out self
    candidates := filter(allNodes, func(n) { return n.Name != selfName })
    // Fisher-Yates shuffle, take first `fanout`
    shuffle(candidates)
    return candidates[:min(fanout, len(candidates))]
}
```

**Retransmission Strategy**:
- Each node maintains a buffer of recently received gossip messages
- Every `retransmitInterval` (500ms), select new random fanout peers and send
- Stop retransmitting when `seen_count >= max_seen_count`
- Critical events (Dead) use `BroadcastToAll` for reliability

### E. Testing Strategy

**Unit Tests**:
- `internal/gossip/event_test.go` — encode/decode roundtrip, unknown event types
- `internal/gossip/dedup_test.go` — seen count tracking, expiry, threshold behavior
- `internal/gossip/disseminator_test.go` — fanout selection, self-exclusion, deterministic seeding

**Integration Tests**:
- `internal/gossip/server_test.go` — start 3 UDP servers, verify event propagation
- Test dedup: send same event 5 times, verify only processed 3 times
- Test fanout: verify all nodes eventually receive events

### F. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| UDP packet loss | Events not delivered | Retransmission + eventual consistency |
| Out-of-order delivery | Stale state | Idempotent event handling, version vectors |
| Large cluster gossip storm | Network saturation | Fanout limit (3), exponential backoff |
| Split-brain | Conflicting state | Tie-breaker: higher seq_no wins |

---

## Feature 2: Failure Detection + Auto Data Migration

### A. Current State Analysis

**dbeel failure_detector.rs** (`/home/veda/dbeel/src/tasks/failure_detector.rs:1-105`):
- Periodic loop with configurable interval (default 500ms)
- Random node selection: `choose(&mut rng)` from all known nodes
- Health check: TCP `Ping` request to random shard port on target node
- On failure: call `handle_dead_node(node_name)` → broadcast `GossipEvent::Dead` to local shards → gossip to cluster
- Runs as a single background task per node (not per shard)

**dbeel shards.rs handle_dead_node** (`/home/veda/dbeel/src/shards.rs:829-851`):
```rust
async fn handle_dead_node(self: Rc<Self>, node_name: &str) {
    // 1. Remove from nodes map
    // 2. Partition shards: removed vs kept
    // 3. Replace shard list with kept
    // 4. Trigger migrate_data_on_node_removal(removed_shards)
}
```

**dbeel shards.rs migrate_data_on_node_removal** (`/home/veda/dbeel/src/shards.rs:853-924`):
- For each collection with replication_factor > 1:
  - Find the "last owning shard" (the farthest replica in the hash ring)
  - Check if any removed shards are between current shard and last owning shard
  - If yes: compute hash range [start, end] → create `RangeAndAction::SendToShard`
- Spawn migration tasks in background

**dbeel shards.rs migrate_data_on_node_addition** (`/home/veda/dbeel/src/shards.rs:926-1072`):
- For each collection:
  - Step 1: Migrate to closest added shard between self and last owning
  - Step 2: Migrate to added shards between previous and self
  - Step 3: Delete items no longer owned by this shard
  - Step 4: Execute migration actions on background tasks

**dbeel migration.rs** (`/home/veda/dbeel/src/tasks/migration.rs:1-169`):
- `MigrationAction` enum: `SendToShard(ShardConnection)` or `Delete`
- `RangeAndAction`: start hash, end hash, action
- `migrate_actions()`: iterate LSM tree with hash filter → send entries to target shards
- Uses `tree.iter_filter()` with hash-based predicate
- Sends `ShardEvent::Set` messages to target shards
- Supports both local (channel) and remote (TCP) targets

**Magnitude coordinator.go** (`/home/veda/Desktop/vector_db/internal/cluster/coordinator.go:231-252`):
- Already has `CheckHealth()` — evaluates nodes based on heartbeat timestamps
- State transitions: Alive → Suspect (heartbeat timeout) → Dead (dead timeout)
- Removes dead nodes from hash ring
- **Missing**: No failure detection goroutine, no ping mechanism, no data migration

### B. Architecture Design

**New package**: `internal/failure/`

```
internal/failure/
├── detector.go     — Failure detection loop
├── ping.go         — Health check protocol
├── migration.go    — Data migration engine
├── state.go        — Node state machine
└── config.go       — Failure detection configuration
```

**New package**: `internal/migration/`

```
internal/migration/
├── engine.go       — Migration orchestrator
├── iterator.go     — Vector iteration by hash range
├── stream.go       — Streaming protocol for vector transfer
├── rollback.go     — Rollback strategy
└── config.go       — Migration configuration
```

**State Machine**:
```
Alive ──[miss 1 heartbeat]──> Suspect ──[miss N heartbeats]──> Dead
  ^                                                              |
  └──────────────[receive Alive gossip]──────────────────────────┘
```

**Goroutines**:
1. `failureDetector` — periodic ping loop, selects random nodes
2. `migrationEngine` — processes migration queue, streams vectors

### C. Implementation Steps

#### Phase 1: Foundation

**Step 1.1**: Create `internal/failure/config.go`
```go
type Config struct {
    Interval          time.Duration `yaml:"interval"`          // ping interval (default: 500ms)
    SuspectThreshold  int           `yaml:"suspectThreshold"`  // missed pings → suspect (default: 1)
    DeadThreshold     int           `yaml:"deadThreshold"`     // missed pings → dead (default: 3)
    PingTimeout       time.Duration `yaml:"pingTimeout"`       // TCP ping timeout (default: 2s)
}
```

**Step 1.2**: Create `internal/failure/state.go`
- `NodeHealthState` struct: nodeID, state, missedPings, lastPingTime
- `StateTracker` struct: `map[string]*NodeHealthState` with mutex
- `RecordPing(nodeID)` — reset missed pings, set Alive
- `RecordMiss(nodeID)` — increment missed pings, update state
- `GetState(nodeID) NodeState` — return current state

**Step 1.3**: Add `Failure` section to `internal/config/config.go`

#### Phase 2: Core Logic

**Step 2.1**: Create `internal/failure/ping.go`
- `PingNode(ctx context.Context, address string, timeout time.Duration) error`
- TCP connect → send `{"type":"ping"}` → expect `{"type":"pong"}` → close
- Returns error on timeout, connection refused, or wrong response

**Step 2.2**: Create `internal/failure/detector.go`
- `Detector` struct: coordinator reference, state tracker, config, gossip client
- `Start(ctx context.Context)` — main loop:
  ```
  for each tick:
      1. Select random alive node
      2. Ping node
      3. If ping fails:
         a. Increment missed pings
         b. If missed >= deadThreshold:
            - Call coordinator.LeaveNode(nodeID)
            - Broadcast GossipEvent::Dead
            - Trigger migration
         c. If missed >= suspectThreshold:
            - Mark as Suspect in state tracker
      4. If ping succeeds:
         - Reset missed pings
         - If was Suspect, mark Alive
  ```
- `Stop()` — cancel context, wait for goroutine exit

**Step 2.3**: Create `internal/migration/engine.go`
- `Engine` struct: coordinator reference, collection manager, config
- `MigrateOnNodeRemoval(removedNodeID string)` — compute affected collections and hash ranges
- `MigrateOnNodeAddition(addedNodeID string)` — compute new ownership, migrate relevant vectors
- `MigrationTask` struct: collectionID, hashRange, targetNode, action (send/delete)

**Step 2.4**: Create `internal/migration/iterator.go`
- `VectorIterator` interface for iterating vectors by hash range
- For Flat index: iterate `AllVectors()`, filter by hash of vector ID
- For HNSW index: iterate `idToNode` map, filter by hash of vector ID
- For IVF index: iterate all clusters, filter by hash of vector ID
- `HashRange` struct: Start, End uint32
- `InRange(hash uint32, r HashRange) bool` — handles wraparound

**Step 2.5**: Create `internal/migration/stream.go`
- `StreamVectors(ctx, collectionID, targetAddress, hashRange) error`
- Batch vectors into messages (100 vectors per batch)
- Send via TCP with length-prefixed JSON framing
- Receive acknowledgment per batch
- Handle backpressure via flow control

**Step 2.6**: Create `internal/migration/rollback.go`
- `RollbackPlan` struct: list of operations to undo on failure
- On migration failure: delete partially transferred vectors on target
- On migration failure: restore removed vectors on source (if still present)
- Log rollback actions for debugging

#### Phase 3: Integration

**Step 3.1**: Wire failure detector into `cmd/server/main.go`
- Create `failure.Detector` after Coordinator and gossip server
- Start detector goroutine
- On shutdown: stop detector

**Step 3.2**: Wire migration engine into Coordinator
- On `CheckHealth()` detecting Dead state: call `migration.MigrateOnNodeRemoval`
- On gossip `Alive` event: call `migration.MigrateOnNodeAddition`

**Step 3.3**: Modify `internal/collection/collection.go`
- Add `ExportVectorsByHashRange(collectionID string, r HashRange) ([]VectorBatch, error)`
- Add `ImportVectorsBatch(collectionID string, batch VectorBatch) error`
- Add `DeleteVectorsByHashRange(collectionID string, r HashRange) error`

### D. Data Migration Design

**Hash Range Iteration by Index Type**:

| Index Type | Iteration Method | Complexity |
|------------|-----------------|------------|
| Flat | `AllVectors()` → filter by `hash(vectorID)` | O(n) |
| HNSW | Iterate `idToNode` map → filter by `hash(vectorID)` | O(n) |
| IVF | Iterate all clusters → all vectors → filter | O(n) |
| SPANN | Iterate posting lists → filter | O(n) |

**Note**: All index types currently store vectors in-memory. Migration requires adding an `ExportVectors()` method to the `Index` interface:

```go
// Add to index/index.go
type VectorExporter interface {
    ExportVectors() []ExportedVector
}

type ExportedVector struct {
    ID     uint64
    Vector []float32
}
```

**Streaming Protocol**:
```
Source Node                              Target Node
    |                                         |
    |──── BatchRequest(collectionID) ────────>|
    |<───── Ack(collectionID, ready) ─────────|
    |                                         |
    |──── VectorBatch(vectors[0:100]) ───────>|
    |<───── Ack(batch_id=0, count=100) ───────|
    |                                         |
    |──── VectorBatch(vectors[100:200]) ──────>|
    |<───── Ack(batch_id=1, count=100) ───────|
    |                                         |
    |──── MigrationComplete(collectionID) ───>|
    |<───── Ack(collectionID, verified) ──────|
```

**Concurrent Reads During Migration**:
- Source node continues serving reads during migration
- Vector is only deleted from source AFTER target confirms receipt
- Two-phase approach:
  1. **Copy phase**: Stream vectors to target. Source is authoritative.
  2. **Switch phase**: After all vectors transferred, update hash ring. Target becomes authoritative.
  3. **Cleanup phase**: Delete migrated vectors from source.

**Rollback Strategy**:
1. **Before migration starts**: Snapshot vector count on source and target
2. **During migration**: Track count of vectors sent vs acknowledged
3. **On failure**:
   - If < 50% transferred: delete partial data on target, keep source intact
   - If >= 50% transferred: complete migration from checkpoint, then clean source
4. **On timeout**: Retry from last checkpoint (batch granularity)

### E. Testing Strategy

**Unit Tests**:
- `internal/failure/state_test.go` — state transitions, threshold behavior
- `internal/failure/ping_test.go` — success, timeout, connection refused
- `internal/migration/iterator_test.go` — hash range filtering, wraparound
- `internal/migration/stream_test.go` — batch encoding, flow control

**Integration Tests**:
- 3-node cluster simulation: kill node 3, verify data migrates to nodes 1 and 2
- Verify read availability during migration
- Verify migration completes within timeout

**Chaos Tests**:
- Kill node during migration → verify rollback
- Network partition during migration → verify consistency
- Concurrent writes during migration → verify no data loss

### F. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Migration overwhelms network | Latency spike | Rate limiting, batch size tuning |
| Split-brain during partition | Data divergence | Quorum-based writes (W+R > N) |
| Migration loop (node flapping) | CPU waste | Exponential backoff on re-join |
| Large collection migration | OOM | Streaming with bounded buffer |

---

## Feature 3: Shard-Level Routing

### A. Current State Analysis

**dbeel shards.rs consistent hashing** (`/home/veda/dbeel/src/shards.rs:95-109`):
```rust
pub fn hash_string(s: &str) -> std::io::Result<u32> {
    murmur3_32(&mut std::io::Cursor::new(s), 0)
}

fn is_between(item: u32, start: u32, end: u32) -> bool {
    if end < start {
        item >= end || item < start
    } else {
        item >= start && item < end
    }
}
```

**dbeel shards.rs routing** (`/home/veda/dbeel/src/shards.rs:586-618`):
- `owns_key(hash, replica_index)` — checks if this shard owns the key at the given replica index
- Hash ring is sorted with this shard's hash as the starting point
- Walks ring in reverse to find replica ownership

**Magnitude hash.go** (`/home/veda/Desktop/vector_db/internal/cluster/hash.go:1-174`):
- Already has `HashRing` with SHA-256 based consistent hashing
- `GetNode(key)` — returns single owner
- `GetNodes(key, n)` — returns N distinct owners for replication
- Uses virtual nodes (150 per physical node)

**Magnitude rendezvous.go** (`/home/veda/Desktop/vector_db/internal/cluster/rendezvous.go:1-143`):
- Already has `RendezvousRouter` as alternative routing strategy
- `Route(key)` — returns single owner
- `RouteN(key, n)` — returns top-N owners

**Magnitude coordinator.go** (`/home/veda/Desktop/vector_db/internal/cluster/coordinator.go:220-229`):
- `RouteCollection(collectionID)` — delegates to HashRing
- `RouteCollectionReplicas(collectionID)` — delegates to HashRing.GetNodes

**Missing**: 
- No per-vector routing (only collection-level)
- No shard-aware API handlers
- No request forwarding to remote nodes
- No read/write splitting for replicas

### B. Architecture Design

**New package**: `internal/routing/`

```
internal/routing/
├── router.go       — Request routing logic
├── forwarder.go    — Request forwarding to remote nodes
├── replica.go      — Replica-aware read/write routing
├── consistency.go  — Consistency level enforcement
└── config.go       — Routing configuration
```

**Routing Levels**:
1. **Collection-level**: Which node owns a collection (existing)
2. **Shard-level**: Which node owns a specific vector ID (new)
3. **Replica-level**: Which replicas to write to / read from (new)

**Consistency Levels** (inspired by Dynamo/Cassandra):
```go
type ConsistencyLevel int

const (
    CLOne    ConsistencyLevel = iota // Write/read to 1 node
    CLQuorum                         // Write/read to majority
    CLAll                           // Write/read to all replicas
)
```

### C. Implementation Steps

#### Phase 1: Foundation

**Step 1.1**: Create `internal/routing/config.go`
```go
type Config struct {
    WriteConsistency  ConsistencyLevel `yaml:"writeConsistency"`  // default: CLQuorum
    ReadConsistency   ConsistencyLevel `yaml:"readConsistency"`   // default: CLOne
    ForwardTimeout    time.Duration    `yaml:"forwardTimeout"`    // default: 5s
    MaxForwards       int              `yaml:"maxForwards"`       // max hop count (default: 3)
}
```

**Step 1.2**: Create `internal/routing/router.go`
- `Router` struct: coordinator reference, config
- `RouteVector(collectionID string, vectorID uint64) (primary string, replicas []string)`
  - Hash: `hash(collectionID + ":" + strconv.FormatUint(vectorID, 10))`
  - Use HashRing to get N nodes
- `RouteCollection(collectionID string) (primary string, replicas []string)`
  - Delegates to coordinator

**Step 1.3**: Add `VectorID` routing to `internal/cluster/hash.go`
- `GetNodeForVector(collectionID string, vectorID uint64) string`
- `GetNodesForVector(collectionID string, vectorID uint64, n int) []string`

#### Phase 2: Core Logic

**Step 2.1**: Create `internal/routing/forwarder.go`
- `Forwarder` struct: HTTP client pool, config
- `ForwardInsert(ctx, targetNode, collectionID, ids, vectors, metadata) error`
- `ForwardSearch(ctx, targetNode, collectionID, query, k, nprobe) ([]SearchResult, error)`
- `ForwardDelete(ctx, targetNode, collectionID, vectorID) error`
- Connection pooling: reuse HTTP clients per target node
- Timeout handling: respect `forwardTimeout`

**Step 2.2**: Create `internal/routing/replica.go`
- `ReplicaRouter` struct: router, forwarder, config
- `WriteWithConsistency(ctx, collectionID, vectorID, op, consistency) error`
  1. Get replicas via router
  2. Send to min(consistency, len(replicas)) nodes in parallel
  3. Wait for required acks
  4. Fire-and-forget to remaining replicas
- `ReadWithConsistency(ctx, collectionID, vectorID, k, consistency) ([]SearchResult, error)`
  1. Get replicas via router
  2. Send search to min(consistency, len(replicas)) nodes in parallel
  3. Merge results (take top-k by score)
  4. Return merged results

**Step 2.3**: Create `internal/routing/consistency.go`
- `QuorumSize(replicationFactor int) int` — returns `replicationFactor/2 + 1`
- `WaitForAcks(results chan error, required int) (int, error)` — collect acks, return count and first error

#### Phase 3: Integration

**Step 3.1**: Modify `internal/api/handlers.go` — Insert handler
```go
func (h *Handler) InsertVectors(w http.ResponseWriter, r *http.Request) {
    // ... existing validation ...
    
    // Route each vector to its owning node
    for i, id := range req.IDs {
        targetNodes := h.router.RouteVector(collectionID, id)
        if isLocalNode(targetNodes[0]) {
            // Insert locally
            h.manager.InsertVectors(ctx, collectionID, []uint64{id}, [][]float32{req.Vectors[i]}, ...)
        } else {
            // Forward to primary owner
            h.forwarder.ForwardInsert(ctx, targetNodes[0], collectionID, ...)
        }
    }
}
```

**Step 3.2**: Modify `internal/api/handlers.go` — Search handler
```go
func (h *Handler) SearchVectors(w http.ResponseWriter, r *http.Request) {
    // Search is collection-level (not vector-level)
    // Route to collection owner
    targetNode := h.router.RouteCollection(collectionID)
    if isLocalNode(targetNode) {
        results, err := h.manager.SearchVectors(...)
    } else {
        results, err = h.forwarder.ForwardSearch(ctx, targetNode, ...)
    }
}
```

**Step 3.3**: Modify `internal/api/handlers.go` — Delete handler
```go
func (h *Handler) DeleteVector(w http.ResponseWriter, r *http.Request) {
    targetNodes := h.router.RouteVector(collectionID, vectorID)
    if isLocalNode(targetNodes[0]) {
        h.manager.DeleteVector(...)
    } else {
        h.forwarder.ForwardDelete(ctx, targetNodes[0], ...)
    }
}
```

**Step 3.4**: Add node identity to Handler
- `Handler` struct gets `nodeID string` and `router *routing.Router`
- `isLocalNode(nodeID string) bool` — checks if target is self

**Step 3.5**: Wire routing into `cmd/server/main.go`
- Create `routing.Router` with coordinator reference
- Create `routing.Forwarder` with HTTP client pool
- Pass router and forwarder to API handler constructor

### D. Testing Strategy

**Unit Tests**:
- `internal/routing/router_test.go` — hash distribution, consistency
- `internal/routing/forwarder_test.go` — HTTP forwarding with mock servers
- `internal/routing/consistency_test.go` — quorum calculation, ack waiting

**Integration Tests**:
- 3-node cluster: insert vector → verify it lands on correct node
- Search across nodes: verify results are merged correctly
- Delete forwarded to correct node

**Load Tests**:
- Verify even distribution of vectors across nodes
- Measure forwarding latency overhead
- Verify no hot spots in hash ring

### E. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Forwarding adds latency | p99 increase | Connection pooling, local-first routing |
| Hash ring rebalance storm | Thundering herd | Gradual rebalance, rate limiting |
| Inconsistent reads during migration | Stale results | Read-repair, version vectors |
| Circular forwarding | Infinite loop | MaxForwards header, loop detection |

---

## Cross-Cutting Concerns

### Integration with Existing Code

#### cmd/server/main.go Changes

```go
// After step 8 (Collection Manager initialization):

// ── 8b. Initialize Cluster Coordinator ──────────────────────────
if cfg.Cluster.Enabled {
    coordinator = cluster.NewCoordinator(cluster.CoordinatorConfig{
        NodeID:            cfg.Cluster.NodeID,
        ReplicationFactor: cfg.Cluster.ReplicationFactor,
        VirtualNodes:      cfg.Cluster.VirtualNodes,
        HeartbeatTimeout:  cfg.Cluster.HeartbeatTimeout,
        DeadTimeout:       cfg.Cluster.DeadTimeout,
    })
    
    // ── 8c. Start Gossip Server ────────────────────────────────
    gossipServer = gossip.NewServer(gossip.Config{...}, coordinator, mgr)
    go gossipServer.Start(ctx)
    
    // ── 8d. Start Failure Detector ─────────────────────────────
    failureDetector = failure.NewDetector(coordinator, failure.Config{...})
    go failureDetector.Start(ctx)
    
    // ── 8e. Start Migration Engine ─────────────────────────────
    migrationEngine = migration.NewEngine(coordinator, mgr)
    
    // ── 8f. Start Inter-node RPC Server ────────────────────────
    rpcServer = cluster.NewRPCServer(coordinator, mgr)
    go rpcServer.Start(ctx)
    
    // ── 8g. Announce self ──────────────────────────────────────
    gossipServer.Broadcast(gossip.GossipEvent{
        Type: gossip.GossipAlive,
        NodeMetadata: &gossip.NodeMetadata{Name: cfg.Cluster.NodeID, ...},
    })
}

// ... existing HTTP server setup ...

// In signal handling (SIGHUP):
// Hot-reload gossip config, failure detection intervals

// In graceful shutdown:
// Step 2.5: Announce Dead, stop gossip, stop failure detector
```

#### Config Changes (config.yaml)

```yaml
cluster:
  enabled: false                    # set to true for distributed mode
  nodeID: "node-1"                  # unique node identifier
  replicationFactor: 3
  virtualNodes: 150
  
  gossip:
    port: 30000
    fanout: 3
    maxSeenCount: 3
    seenExpiry: 30s
    retransmitInterval: 500ms
  
  failure:
    interval: 500ms
    suspectThreshold: 1
    deadThreshold: 3
    pingTimeout: 2s
  
  routing:
    writeConsistency: "quorum"      # one | quorum | all
    readConsistency: "one"
    forwardTimeout: 5s
    maxForwards: 3
  
  migration:
    batchSize: 100
    rateLimit: 10MB/s
    checkpointInterval: 10s
```

#### Collection Manager Changes

```go
// internal/collection/collection.go additions:

// ExportVectors returns all vectors in the collection for migration.
func (m *Manager) ExportVectors(collectionID string) ([]ExportedVector, error) {
    col := m.collections[collectionID]
    // Use index-specific export method
    if exporter, ok := col.idx.(index.VectorExporter); ok {
        vectors := exporter.ExportVectors()
        // Add metadata
        for i := range vectors {
            vectors[i].Metadata = col.vectorMeta[vectors[i].ID]
        }
        return vectors, nil
    }
    return nil, fmt.Errorf("index does not support export")
}

// ImportVectors adds vectors from migration (skips WAL, as migration is idempotent).
func (m *Manager) ImportVectors(collectionID string, vectors []ExportedVector) error {
    col := m.collections[collectionID]
    col.mu.Lock()
    defer col.mu.Unlock()
    
    for _, v := range vectors {
        if err := col.idx.Insert(v.ID, v.Vector); err != nil {
            // Idempotent: skip duplicates
            continue
        }
        if v.Metadata != nil {
            col.vectorMeta[v.ID] = metadata.VectorMetadata(v.Metadata)
        }
    }
    return nil
}
```

#### Index Interface Extension

```go
// internal/index/index.go additions:

// VectorExporter is an optional interface that indexes can implement
// to support vector enumeration for migration.
type VectorExporter interface {
    ExportVectors() []ExportedVector
}

type ExportedVector struct {
    ID       uint64
    Vector   []float32
    Metadata map[string]any
}
```

### Complete Dependency Graph

```
gossip/event.go          ← no deps
gossip/config.go         ← no deps
gossip/dedup.go          ← gossip/event.go
gossip/disseminator.go   ← gossip/event.go
gossip/server.go         ← gossip/event.go, gossip/dedup.go, gossip/disseminator.go
gossip/event_handler.go  ← gossip/event.go

failure/config.go        ← no deps
failure/state.go         ← cluster/coordinator.go
failure/ping.go          ← no deps
failure/detector.go      ← failure/state.go, failure/ping.go, gossip (for Dead broadcast)

migration/config.go      ← no deps
migration/iterator.go    ← index/index.go, cluster/hash.go
migration/stream.go      ← cluster/client.go
migration/engine.go      ← migration/iterator.go, migration/stream.go, collection/manager
migration/rollback.go    ← migration/engine.go

routing/config.go        ← no deps
routing/router.go        ← cluster/coordinator.go, cluster/hash.go
routing/forwarder.go     ← cluster/client.go
routing/replica.go       ← routing/router.go, routing/forwarder.go
routing/consistency.go   ← no deps

cluster/client.go        ← no deps (TCP client for inter-node RPC)
cluster/rpc_server.go    ← collection/manager
```

### Implementation Order

```
Phase 1 (Foundation):
  1. gossip/event.go + gossip/config.go
  2. gossip/dedup.go
  3. failure/config.go + failure/state.go
  4. routing/config.go
  5. Add Cluster config to config.go
  6. Add VectorExporter to index/index.go

Phase 2 (Core - Gossip):
  7. gossip/disseminator.go
  8. gossip/server.go
  9. gossip/event_handler.go
  10. cluster/client.go (inter-node RPC)
  11. cluster/rpc_server.go

Phase 3 (Core - Failure):
  12. failure/ping.go
  13. failure/detector.go

Phase 4 (Core - Migration):
  14. migration/iterator.go
  15. migration/stream.go
  16. migration/engine.go
  17. migration/rollback.go

Phase 5 (Core - Routing):
  18. routing/router.go
  19. routing/forwarder.go
  20. routing/replica.go
  21. routing/consistency.go

Phase 6 (Integration):
  22. Add ExportVectors/ImportVectors to collection.go
  23. Add VectorExporter to each index type
  24. Wire gossip into cmd/server/main.go
  25. Wire failure detector into cmd/server/main.go
  26. Wire routing into API handlers
  27. Wire migration into coordinator events

Phase 7 (Testing):
  28. Unit tests for all new packages
  29. Integration tests: multi-node simulation
  30. Chaos tests: failure during migration
```

### Estimated Effort

| Feature | Files | Lines of Code | Effort |
|---------|-------|---------------|--------|
| Gossip Protocol | 7 | ~800 | 3-4 days |
| Failure Detection | 5 | ~500 | 2-3 days |
| Data Migration | 5 | ~700 | 3-4 days |
| Shard-Level Routing | 5 | ~600 | 2-3 days |
| Integration | 4 files modified | ~300 | 2-3 days |
| Testing | 15 test files | ~1500 | 4-5 days |
| **Total** | **~26 new files** | **~4400** | **16-22 days** |

### Key Design Decisions

1. **UDP for gossip, TCP for RPC**: UDP is fire-and-forget (low overhead for metadata dissemination). TCP is reliable for vector transfer and request forwarding.

2. **JSON over bincode**: Go's `encoding/json` is standard. bincode is Rust-specific. JSON adds ~2x overhead on wire but is debuggable and interoperable.

3. **Collection-level vs vector-level routing**: Collection-level is simpler but less balanced. Vector-level provides better distribution but adds complexity. **Recommendation**: Start with collection-level, add vector-level later.

4. **Streaming migration**: Don't load all vectors into memory. Stream in batches of 100. Use backpressure to avoid overwhelming the target node.

5. **Idempotent operations**: All gossip events, migration operations, and routing must be idempotent. Duplicate messages should have no effect.

6. **No split-brain resolution in v1**: Use simple majority quorum. Split-brain scenarios require more sophisticated protocols (Raft/Paxos) which are out of scope for Tier 3.
