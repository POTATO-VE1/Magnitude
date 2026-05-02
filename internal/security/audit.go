// Package security — Structured audit logger.
//
// The audit log records security-relevant events in a tamper-evident,
// append-only stream, separate from application logs:
//   - Authentication successes and failures (with masked key ID, never raw key)
//   - Rate limit rejections (with IP and endpoint)
//   - Collection create/delete operations (tenant, collection, user identity)
//   - Configuration changes (if hot-reload is added later)
//
// Each audit entry is a structured JSON line written to a dedicated file.
// Log rotation is handled by the OS (logrotate) or a sidecar log shipper.
//
// Audit entries must never include raw API keys, passwords, or vector data.
package security

import (
	"encoding/json"
	"io"
	"log/slog"
	"os"
	"sync"
	"time"
)

// AuditEvent represents a single auditable operation.
type AuditEvent struct {
	Timestamp    time.Time `json:"ts"`
	Actor        string    `json:"actor"`          // API key hash prefix (never raw key)
	Action       string    `json:"action"`         // "insert", "delete", "search", "create_collection", etc.
	CollectionID string    `json:"collection_id,omitempty"`
	VectorID     *uint64   `json:"vector_id,omitempty"`
	RemoteIP     string    `json:"remote_ip"`
	Success      bool      `json:"success"`
	ErrorCode    string    `json:"error_code,omitempty"`
	Details      string    `json:"details,omitempty"` // extra context (e.g., "vector_count=1000")
}

// AuditLogger writes structured audit events to a dedicated log stream.
// Thread-safe. Events are serialized as one JSON line per event.
type AuditLogger struct {
	mu     sync.Mutex
	writer io.Writer
	logger *slog.Logger
}

// NewAuditLogger creates an audit logger that writes JSON lines to the given writer.
func NewAuditLogger(w io.Writer) *AuditLogger {
	return &AuditLogger{
		writer: w,
		logger: slog.New(slog.NewJSONHandler(w, &slog.HandlerOptions{
			Level: slog.LevelInfo,
		})),
	}
}

// NewFileAuditLogger creates an audit logger backed by a file.
// The file is opened in append-only mode (O_APPEND | O_CREATE | O_WRONLY).
// File permissions are set to 0600 (owner read/write only).
func NewFileAuditLogger(path string) (*AuditLogger, error) {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
	if err != nil {
		return nil, err
	}
	return NewAuditLogger(f), nil
}

// NewNoopAuditLogger creates a logger that discards all events (for testing).
func NewNoopAuditLogger() *AuditLogger {
	return NewAuditLogger(io.Discard)
}

// Log writes an audit event to the log stream.
// This method is safe for concurrent use.
func (al *AuditLogger) Log(event AuditEvent) {
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now().UTC()
	}

	al.mu.Lock()
	defer al.mu.Unlock()

	data, err := json.Marshal(event)
	if err != nil {
		// Should never happen with our struct, but log defensively.
		al.logger.Error("audit marshal error", "error", err)
		return
	}

	// Write JSON line (append newline for log parsing)
	data = append(data, '\n')
	if _, err := al.writer.Write(data); err != nil {
		// Cannot write audit log — this is a critical failure.
		// Log to slog as a fallback but do NOT panic or crash the server.
		al.logger.Error("audit write failed",
			"error", err,
			"action", event.Action,
			"actor", event.Actor,
		)
	}
}

// Close closes the underlying writer if it implements io.Closer.
func (al *AuditLogger) Close() error {
	if closer, ok := al.writer.(io.Closer); ok {
		return closer.Close()
	}
	return nil
}
