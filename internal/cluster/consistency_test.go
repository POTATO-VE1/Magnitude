package cluster

import (
	"testing"
)

func TestConsistencyLevel_String(t *testing.T) {
	tests := []struct {
		level ConsistencyLevel
		want  string
	}{
		{ConsistencyOne, "one"},
		{ConsistencyQuorum, "quorum"},
		{ConsistencyAll, "all"},
		{ConsistencyLevel(99), "unknown"},
	}
	for _, tt := range tests {
		if got := tt.level.String(); got != tt.want {
			t.Errorf("ConsistencyLevel(%d).String() = %q, want %q", tt.level, got, tt.want)
		}
	}
}

func TestParseConsistencyLevel(t *testing.T) {
	tests := []struct {
		input   string
		want    ConsistencyLevel
		wantErr bool
	}{
		{"one", ConsistencyOne, false},
		{"quorum", ConsistencyQuorum, false},
		{"all", ConsistencyAll, false},
		{"", ConsistencyOne, false}, // default
		{"invalid", 0, true},
	}
	for _, tt := range tests {
		got, err := ParseConsistencyLevel(tt.input)
		if (err != nil) != tt.wantErr {
			t.Errorf("ParseConsistencyLevel(%q) error = %v, wantErr %v", tt.input, err, tt.wantErr)
			continue
		}
		if got != tt.want {
			t.Errorf("ParseConsistencyLevel(%q) = %d, want %d", tt.input, got, tt.want)
		}
	}
}

func TestResolveConsistency(t *testing.T) {
	tests := []struct {
		level ConsistencyLevel
		rf    int
		want  int
	}{
		{ConsistencyOne, 1, 1},
		{ConsistencyOne, 3, 1},
		{ConsistencyOne, 5, 1},
		{ConsistencyQuorum, 1, 1},
		{ConsistencyQuorum, 3, 2},
		{ConsistencyQuorum, 5, 3},
		{ConsistencyAll, 1, 1},
		{ConsistencyAll, 3, 3},
		{ConsistencyAll, 5, 5},
	}
	for _, tt := range tests {
		got := ResolveConsistency(tt.level, tt.rf)
		if got != tt.want {
			t.Errorf("ResolveConsistency(%v, %d) = %d, want %d", tt.level, tt.rf, got, tt.want)
		}
	}
}

func TestResolveConsistency_ClampsToRF(t *testing.T) {
	// If requested consistency exceeds RF, clamp to RF
	got := ResolveConsistency(ConsistencyAll, 1)
	if got != 1 {
		t.Errorf("ResolveConsistency(All, 1) = %d, want 1", got)
	}
}
