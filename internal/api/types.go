// Package api — Additional response types for the VectorDB HTTP API.
//
// Request types are defined inline in handlers.go alongside the handlers that use them.
// Response types shared across multiple handlers are defined here.
package api

// SearchResultItem is a single result in a search response.
type SearchResultItem struct {
	ID       uint64         `json:"id"`
	Distance float32        `json:"distance"`
	Score    float32        `json:"score"`
	Metadata map[string]any `json:"metadata,omitempty"`
}
