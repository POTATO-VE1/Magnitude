// Package metadata — filter expression parser and evaluator for hybrid search.
//
// Filter expressions allow combining vector similarity search with metadata
// predicates. This is ChromaDB's "where" clause equivalent.
//
// Supported operators:
//   $eq  — equality
//   $ne  — not equal
//   $gt  — greater than
//   $gte — greater than or equal
//   $lt  — less than
//   $lte — less than or equal
//   $in  — value in set
//   $nin — value not in set
//   $and — logical AND of sub-filters
//   $or  — logical OR of sub-filters
//
// Example filter (JSON):
//
//	{"category": {"$eq": "science"}, "year": {"$gte": 2020}}
//
// This is implicitly AND-ed: category == "science" AND year >= 2020.
package metadata

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Filter represents a parsed filter expression tree.
type Filter struct {
	// AND combines multiple field conditions (implicit top-level)
	AND []FieldFilter
	// OR is populated when $or is used
	OR []Filter
}

// FieldFilter is a single field comparison.
type FieldFilter struct {
	Field    string
	Operator string
	Value    any
	Values   []any // for $in/$nin
}

// VectorMetadata is a key-value map attached to each vector.
type VectorMetadata map[string]any

// ParseFilter parses a JSON filter map into a Filter struct.
// Input: {"field1": {"$op": value}, "$or": [...]}
func ParseFilter(raw map[string]any) (*Filter, error) {
	if len(raw) == 0 {
		return nil, nil
	}

	f := &Filter{}

	for key, val := range raw {
		switch key {
		case "$and":
			arr, ok := val.([]any)
			if !ok {
				return nil, fmt.Errorf("filter: $and must be an array")
			}
			for _, item := range arr {
				subMap, ok := item.(map[string]any)
				if !ok {
					return nil, fmt.Errorf("filter: $and elements must be objects")
				}
				subFilter, err := ParseFilter(subMap)
				if err != nil {
					return nil, err
				}
				f.AND = append(f.AND, subFilter.AND...)
			}

		case "$or":
			arr, ok := val.([]any)
			if !ok {
				return nil, fmt.Errorf("filter: $or must be an array")
			}
			for _, item := range arr {
				subMap, ok := item.(map[string]any)
				if !ok {
					return nil, fmt.Errorf("filter: $or elements must be objects")
				}
				subFilter, err := ParseFilter(subMap)
				if err != nil {
					return nil, err
				}
				f.OR = append(f.OR, *subFilter)
			}

		default:
			// Field condition: {"field": {"$op": value}} or {"field": value} (shorthand for $eq)
			switch v := val.(type) {
			case map[string]any:
				for op, opVal := range v {
					ff := FieldFilter{
						Field:    key,
						Operator: op,
					}
					if op == "$in" || op == "$nin" {
						arr, ok := opVal.([]any)
						if !ok {
							return nil, fmt.Errorf("filter: %s requires an array value", op)
						}
						ff.Values = arr
					} else {
						ff.Value = opVal
					}
					f.AND = append(f.AND, ff)
				}
			default:
				// Shorthand: {"field": value} → {"field": {"$eq": value}}
				f.AND = append(f.AND, FieldFilter{
					Field:    key,
					Operator: "$eq",
					Value:    val,
				})
			}
		}
	}

	return f, nil
}

// ParseFilterJSON parses a JSON byte slice into a Filter.
func ParseFilterJSON(data []byte) (*Filter, error) {
	if len(data) == 0 {
		return nil, nil
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("filter: invalid JSON: %w", err)
	}
	return ParseFilter(raw)
}

// Match evaluates the filter against a metadata map.
// Returns true if the metadata satisfies all filter conditions.
func (f *Filter) Match(meta VectorMetadata) bool {
	if f == nil {
		return true
	}

	// All AND conditions must match
	for _, ff := range f.AND {
		if !ff.Match(meta) {
			return false
		}
	}

	// If OR conditions exist, at least one must match
	if len(f.OR) > 0 {
		for _, orFilter := range f.OR {
			if orFilter.Match(meta) {
				return true
			}
		}
		return false
	}

	return true
}

// Match evaluates a single field filter against metadata.
func (ff *FieldFilter) Match(meta VectorMetadata) bool {
	val, exists := meta[ff.Field]

	switch ff.Operator {
	case "$eq":
		if !exists {
			return false
		}
		return compareEqual(val, ff.Value)

	case "$ne":
		if !exists {
			return true // non-existent field != anything
		}
		return !compareEqual(val, ff.Value)

	case "$gt":
		if !exists {
			return false
		}
		return compareNumeric(val, ff.Value) > 0

	case "$gte":
		if !exists {
			return false
		}
		return compareNumeric(val, ff.Value) >= 0

	case "$lt":
		if !exists {
			return false
		}
		return compareNumeric(val, ff.Value) < 0

	case "$lte":
		if !exists {
			return false
		}
		return compareNumeric(val, ff.Value) <= 0

	case "$in":
		if !exists {
			return false
		}
		for _, v := range ff.Values {
			if compareEqual(val, v) {
				return true
			}
		}
		return false

	case "$nin":
		if !exists {
			return true
		}
		for _, v := range ff.Values {
			if compareEqual(val, v) {
				return false
			}
		}
		return true

	default:
		return false
	}
}

// compareEqual compares two values for equality, handling type coercion.
func compareEqual(a, b any) bool {
	// Handle numeric types (JSON numbers are float64)
	aNum, aIsNum := toFloat64(a)
	bNum, bIsNum := toFloat64(b)
	if aIsNum && bIsNum {
		return aNum == bNum
	}

	// String comparison
	aStr, aIsStr := a.(string)
	bStr, bIsStr := b.(string)
	if aIsStr && bIsStr {
		return aStr == bStr
	}

	// Bool comparison
	aBool, aIsBool := a.(bool)
	bBool, bIsBool := b.(bool)
	if aIsBool && bIsBool {
		return aBool == bBool
	}

	// Fallback: string representation
	return fmt.Sprintf("%v", a) == fmt.Sprintf("%v", b)
}

// compareNumeric compares two numeric values. Returns -1, 0, or 1.
func compareNumeric(a, b any) int {
	aNum, aOk := toFloat64(a)
	bNum, bOk := toFloat64(b)
	if !aOk || !bOk {
		// Try string comparison for non-numeric types
		aStr := fmt.Sprintf("%v", a)
		bStr := fmt.Sprintf("%v", b)
		return strings.Compare(aStr, bStr)
	}
	if aNum < bNum {
		return -1
	}
	if aNum > bNum {
		return 1
	}
	return 0
}

// toFloat64 converts various numeric types to float64.
func toFloat64(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case int32:
		return float64(n), true
	case uint64:
		return float64(n), true
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	default:
		return 0, false
	}
}
