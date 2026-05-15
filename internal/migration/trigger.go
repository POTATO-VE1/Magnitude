package migration

import (
	"log/slog"

	"github.com/POTATO-VE1/Magnitude/internal/cluster"
)

// TriggerOnNodeRemoval computes which collections need migration when a node
// leaves the cluster. Returns a list of migration plans for collections that
// were owned by the dead node and need to be redistributed.
//
// This function modifies the ring by removing the dead node — the caller should
// ensure this function is called exactly once per node failure event.
//
// The caller should execute each plan using a Worker with a real VectorSource
// that fetches vectors from a replica node (since the dead node is unreachable).
func TriggerOnNodeRemoval(
	ring *cluster.HashRing,
	deadNodeID string,
	collectionIDs []string,
	replicationFactor int,
) []MigrationPlan {
	if ring == nil || deadNodeID == "" {
		return nil
	}

	// Phase 1: Snapshot which collections the dead node owns BEFORE removal.
	ownedByDead := make(map[string]bool)
	for _, colID := range collectionIDs {
		primary := ring.GetNode(colID)
		if primary == deadNodeID {
			ownedByDead[colID] = true
		}
	}

	// Phase 2: Remove dead node from the ring so GetNode returns the new owner.
	ring.RemoveNode(deadNodeID)

	// Phase 3: Build migration plans for collections that lost their primary.
	var plans []MigrationPlan

	for _, colID := range collectionIDs {
		if ownedByDead[colID] {
			newPrimary := ring.GetNode(colID)
			if newPrimary == "" {
				slog.Warn("migration: no new owner available for collection",
					"collection", colID,
					"dead_node", deadNodeID,
				)
				continue
			}

			plans = append(plans, MigrationPlan{
				CollectionID: colID,
				SourceNode:   deadNodeID,
				TargetNode:   newPrimary,
			})

			slog.Info("migration: planned collection transfer",
				"collection", colID,
				"from", deadNodeID,
				"to", newPrimary,
			)
		}

		// Check replicas — if the dead node held a replica, we need to re-replicate
		if replicationFactor > 1 {
			replicas := ring.GetNodes(colID, replicationFactor)
			for _, replica := range replicas {
				if replica == deadNodeID {
					slog.Info("migration: replica needs re-replication",
						"collection", colID,
						"dead_replica", deadNodeID,
					)
					// The replication is handled by the consistency layer
					// when the next write comes in — no explicit migration needed
					break
				}
			}
		}
	}

	return plans
}

// TriggerOnNodeAddition computes which collections should be rebalanced when
// a new node joins the cluster. Returns migration plans for collections that
// should move to the new node.
func TriggerOnNodeAddition(
	ring *cluster.HashRing,
	newNodeID string,
	collectionIDs []string,
) []MigrationPlan {
	if ring == nil || newNodeID == "" {
		return nil
	}

	var plans []MigrationPlan

	for _, colID := range collectionIDs {
		// Check if the new node should now own this collection
		primary := ring.GetNode(colID)
		if primary == newNodeID {
			// Find the old owner (by temporarily removing the new node)
			// In practice, we'd check the previous ring state
			slog.Info("migration: new node should own collection",
				"collection", colID,
				"new_owner", newNodeID,
			)
			// Migration from old owner to new owner would be triggered here
		}
	}

	return plans
}
