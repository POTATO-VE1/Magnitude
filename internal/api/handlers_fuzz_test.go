package api

import (
	"encoding/json"
	"testing"
)

// FuzzValidateMetadata fuzzes the metadata validator to ensure it doesn't panic
// on unexpected or malformed metadata inputs.
func FuzzValidateMetadata(f *testing.F) {
	// Seed corpus with valid and edge cases
	f.Add(`{"key": "value"}`)
	f.Add(`{}`)
	f.Add(`{"valid_key": 12345}`)
	f.Add(`{"toolong": "a string that is very long but maybe not too long..."}`)
	f.Add(`{"invalid key format!!!": "value"}`)

	f.Fuzz(func(t *testing.T, raw string) {
		var m map[string]any
		if err := json.Unmarshal([]byte(raw), &m); err != nil {
			// If it's not valid JSON, we skip it because it wouldn't reach the validator
			// as a map[string]any anyway.
			return
		}

		// The validator must never panic, regardless of the map contents.
		// It should only return an error if invalid.
		_ = validateMetadata(m)
	})
}
