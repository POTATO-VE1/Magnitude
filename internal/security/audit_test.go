package security

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAuditLogger_Log(t *testing.T) {
	var buf bytes.Buffer
	logger := NewAuditLogger(&buf)

	ts := time.Date(2025, 1, 15, 10, 30, 0, 0, time.UTC)
	vid := uint64(42)
	logger.Log(AuditEvent{
		Timestamp:    ts,
		Actor:        "sha256:a1b2c3d4",
		Action:       "insert",
		CollectionID: "col-123",
		VectorID:     &vid,
		RemoteIP:     "192.168.1.1",
		Success:      true,
	})

	// Should produce a single JSON line
	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	require.Len(t, lines, 1)

	var event AuditEvent
	require.NoError(t, json.Unmarshal([]byte(lines[0]), &event))
	assert.Equal(t, "insert", event.Action)
	assert.Equal(t, "sha256:a1b2c3d4", event.Actor)
	assert.Equal(t, "col-123", event.CollectionID)
	assert.Equal(t, uint64(42), *event.VectorID)
	assert.True(t, event.Success)
	assert.Equal(t, ts, event.Timestamp)
}

func TestAuditLogger_MultipleEvents(t *testing.T) {
	var buf bytes.Buffer
	logger := NewAuditLogger(&buf)

	for i := 0; i < 5; i++ {
		logger.Log(AuditEvent{
			Action:  "search",
			Actor:   "test-actor",
			Success: true,
		})
	}

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	assert.Len(t, lines, 5, "should produce one JSON line per event")
}

func TestAuditLogger_ErrorEvent(t *testing.T) {
	var buf bytes.Buffer
	logger := NewAuditLogger(&buf)

	logger.Log(AuditEvent{
		Action:    "delete",
		Actor:     "sha256:deadbeef",
		Success:   false,
		ErrorCode: "ErrCollectionNotFound",
		Details:   "collection ID 'nonexistent' not found",
	})

	var event AuditEvent
	require.NoError(t, json.Unmarshal(buf.Bytes(), &event))
	assert.False(t, event.Success)
	assert.Equal(t, "ErrCollectionNotFound", event.ErrorCode)
}

func TestAuditLogger_DefaultTimestamp(t *testing.T) {
	var buf bytes.Buffer
	logger := NewAuditLogger(&buf)

	before := time.Now().UTC()
	logger.Log(AuditEvent{Action: "test"})
	after := time.Now().UTC()

	var event AuditEvent
	require.NoError(t, json.Unmarshal(buf.Bytes(), &event))
	assert.True(t, !event.Timestamp.Before(before) && !event.Timestamp.After(after),
		"default timestamp should be approximately now")
}

func TestNoopAuditLogger(t *testing.T) {
	logger := NewNoopAuditLogger()
	// Should not panic
	logger.Log(AuditEvent{Action: "test"})
	require.NoError(t, logger.Close())
}

func TestAuditLogger_Close(t *testing.T) {
	var buf bytes.Buffer
	logger := NewAuditLogger(&buf)
	require.NoError(t, logger.Close()) // bytes.Buffer doesn't implement Closer, should be nil
}

func TestAuditLogger_ConcurrentWrites(t *testing.T) {
	var buf bytes.Buffer
	logger := NewAuditLogger(&buf)

	done := make(chan struct{})
	for i := 0; i < 100; i++ {
		go func(n int) {
			logger.Log(AuditEvent{
				Action:  "search",
				Actor:   "concurrent-test",
				Details: "goroutine test",
			})
			done <- struct{}{}
		}(i)
	}

	for i := 0; i < 100; i++ {
		<-done
	}

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	assert.Len(t, lines, 100, "all 100 events should be written")

	// Verify each line is valid JSON
	for i, line := range lines {
		var event AuditEvent
		assert.NoError(t, json.Unmarshal([]byte(line), &event),
			"line %d should be valid JSON: %s", i, line)
	}
}
