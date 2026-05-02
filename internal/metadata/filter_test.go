package metadata

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseFilter_Eq(t *testing.T) {
	raw := map[string]any{
		"category": map[string]any{"$eq": "science"},
	}
	f, err := ParseFilter(raw)
	require.NoError(t, err)
	require.Len(t, f.AND, 1)
	assert.Equal(t, "category", f.AND[0].Field)
	assert.Equal(t, "$eq", f.AND[0].Operator)
	assert.Equal(t, "science", f.AND[0].Value)
}

func TestParseFilter_Shorthand(t *testing.T) {
	raw := map[string]any{
		"category": "science",
	}
	f, err := ParseFilter(raw)
	require.NoError(t, err)
	require.Len(t, f.AND, 1)
	assert.Equal(t, "$eq", f.AND[0].Operator)
}

func TestParseFilter_Multiple(t *testing.T) {
	raw := map[string]any{
		"category": "science",
		"year":     map[string]any{"$gte": float64(2020)},
	}
	f, err := ParseFilter(raw)
	require.NoError(t, err)
	assert.Len(t, f.AND, 2)
}

func TestParseFilter_In(t *testing.T) {
	raw := map[string]any{
		"color": map[string]any{"$in": []any{"red", "blue", "green"}},
	}
	f, err := ParseFilter(raw)
	require.NoError(t, err)
	require.Len(t, f.AND, 1)
	assert.Equal(t, "$in", f.AND[0].Operator)
	assert.Len(t, f.AND[0].Values, 3)
}

func TestParseFilter_Or(t *testing.T) {
	raw := map[string]any{
		"$or": []any{
			map[string]any{"color": "red"},
			map[string]any{"color": "blue"},
		},
	}
	f, err := ParseFilter(raw)
	require.NoError(t, err)
	assert.Len(t, f.OR, 2)
}

func TestParseFilter_Nil(t *testing.T) {
	f, err := ParseFilter(nil)
	require.NoError(t, err)
	assert.Nil(t, f)
}

func TestParseFilterJSON(t *testing.T) {
	data := []byte(`{"category": {"$eq": "science"}, "year": {"$gte": 2020}}`)
	f, err := ParseFilterJSON(data)
	require.NoError(t, err)
	assert.Len(t, f.AND, 2)
}

// ── Match tests ─────────────────────────────────────────────────────────────

func TestMatch_Eq(t *testing.T) {
	f := &Filter{AND: []FieldFilter{{Field: "category", Operator: "$eq", Value: "science"}}}
	assert.True(t, f.Match(VectorMetadata{"category": "science"}))
	assert.False(t, f.Match(VectorMetadata{"category": "math"}))
	assert.False(t, f.Match(VectorMetadata{}))
}

func TestMatch_Ne(t *testing.T) {
	f := &Filter{AND: []FieldFilter{{Field: "status", Operator: "$ne", Value: "deleted"}}}
	assert.True(t, f.Match(VectorMetadata{"status": "active"}))
	assert.False(t, f.Match(VectorMetadata{"status": "deleted"}))
	assert.True(t, f.Match(VectorMetadata{})) // non-existent != anything
}

func TestMatch_Gt(t *testing.T) {
	f := &Filter{AND: []FieldFilter{{Field: "year", Operator: "$gt", Value: float64(2020)}}}
	assert.True(t, f.Match(VectorMetadata{"year": float64(2021)}))
	assert.False(t, f.Match(VectorMetadata{"year": float64(2020)}))
	assert.False(t, f.Match(VectorMetadata{"year": float64(2019)}))
}

func TestMatch_Gte(t *testing.T) {
	f := &Filter{AND: []FieldFilter{{Field: "year", Operator: "$gte", Value: float64(2020)}}}
	assert.True(t, f.Match(VectorMetadata{"year": float64(2020)}))
	assert.True(t, f.Match(VectorMetadata{"year": float64(2021)}))
	assert.False(t, f.Match(VectorMetadata{"year": float64(2019)}))
}

func TestMatch_Lt(t *testing.T) {
	f := &Filter{AND: []FieldFilter{{Field: "price", Operator: "$lt", Value: float64(100)}}}
	assert.True(t, f.Match(VectorMetadata{"price": float64(50)}))
	assert.False(t, f.Match(VectorMetadata{"price": float64(100)}))
}

func TestMatch_Lte(t *testing.T) {
	f := &Filter{AND: []FieldFilter{{Field: "price", Operator: "$lte", Value: float64(100)}}}
	assert.True(t, f.Match(VectorMetadata{"price": float64(100)}))
	assert.True(t, f.Match(VectorMetadata{"price": float64(50)}))
	assert.False(t, f.Match(VectorMetadata{"price": float64(101)}))
}

func TestMatch_In(t *testing.T) {
	f := &Filter{AND: []FieldFilter{{Field: "color", Operator: "$in", Values: []any{"red", "blue"}}}}
	assert.True(t, f.Match(VectorMetadata{"color": "red"}))
	assert.True(t, f.Match(VectorMetadata{"color": "blue"}))
	assert.False(t, f.Match(VectorMetadata{"color": "green"}))
	assert.False(t, f.Match(VectorMetadata{}))
}

func TestMatch_Nin(t *testing.T) {
	f := &Filter{AND: []FieldFilter{{Field: "status", Operator: "$nin", Values: []any{"deleted", "archived"}}}}
	assert.True(t, f.Match(VectorMetadata{"status": "active"}))
	assert.False(t, f.Match(VectorMetadata{"status": "deleted"}))
	assert.True(t, f.Match(VectorMetadata{})) // non-existent is not in the list
}

func TestMatch_AND(t *testing.T) {
	f := &Filter{AND: []FieldFilter{
		{Field: "category", Operator: "$eq", Value: "science"},
		{Field: "year", Operator: "$gte", Value: float64(2020)},
	}}
	assert.True(t, f.Match(VectorMetadata{"category": "science", "year": float64(2021)}))
	assert.False(t, f.Match(VectorMetadata{"category": "science", "year": float64(2019)}))
	assert.False(t, f.Match(VectorMetadata{"category": "math", "year": float64(2021)}))
}

func TestMatch_OR(t *testing.T) {
	f := &Filter{
		OR: []Filter{
			{AND: []FieldFilter{{Field: "color", Operator: "$eq", Value: "red"}}},
			{AND: []FieldFilter{{Field: "color", Operator: "$eq", Value: "blue"}}},
		},
	}
	assert.True(t, f.Match(VectorMetadata{"color": "red"}))
	assert.True(t, f.Match(VectorMetadata{"color": "blue"}))
	assert.False(t, f.Match(VectorMetadata{"color": "green"}))
}

func TestMatch_NilFilter(t *testing.T) {
	var f *Filter
	assert.True(t, f.Match(VectorMetadata{"anything": "value"}))
}

func TestMatch_IntConversion(t *testing.T) {
	// JSON numbers are float64, but stored metadata might be int
	f := &Filter{AND: []FieldFilter{{Field: "count", Operator: "$eq", Value: float64(42)}}}
	assert.True(t, f.Match(VectorMetadata{"count": 42}))
	assert.True(t, f.Match(VectorMetadata{"count": float64(42)}))
}

func TestMatch_BoolValues(t *testing.T) {
	f := &Filter{AND: []FieldFilter{{Field: "active", Operator: "$eq", Value: true}}}
	assert.True(t, f.Match(VectorMetadata{"active": true}))
	assert.False(t, f.Match(VectorMetadata{"active": false}))
}
