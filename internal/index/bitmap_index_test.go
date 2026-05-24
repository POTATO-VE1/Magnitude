package index

import (
	"testing"

	"github.com/POTATO-VE1/Magnitude/internal/metadata"
)

func TestMetadataBitmapIndex_SetAndGet(t *testing.T) {
	idx := NewMetadataBitmapIndex()

	// Set metadata for nodes 0-4
	idx.Set("category", "electronics", 0)
	idx.Set("category", "electronics", 1)
	idx.Set("category", "books", 2)
	idx.Set("category", "books", 3)
	idx.Set("category", "electronics", 4)

	// Resolve filter: category == electronics
	filter := &metadata.Filter{
		AND: []metadata.FieldFilter{
			{Field: "category", Operator: "$eq", Value: "electronics"},
		},
	}

	bm := idx.Resolve(filter)
	if bm == nil {
		t.Fatal("expected non-nil bitmap")
	}

	// Should match nodes 0, 1, 4
	if !bm.Test(0) || !bm.Test(1) || !bm.Test(4) {
		t.Errorf("expected bits 0,1,4 set, got: %v", bm)
	}
	if bm.Test(2) || bm.Test(3) {
		t.Errorf("expected bits 2,3 NOT set")
	}
}

func TestMetadataBitmapIndex_AND(t *testing.T) {
	idx := NewMetadataBitmapIndex()

	// Node 0: category=electronics, year=2024
	idx.Set("category", "electronics", 0)
	idx.Set("year", "2024", 0)

	// Node 1: category=electronics, year=2023
	idx.Set("category", "electronics", 1)
	idx.Set("year", "2023", 1)

	// Node 2: category=books, year=2024
	idx.Set("category", "books", 2)
	idx.Set("year", "2024", 2)

	// Filter: category=electronics AND year=2024
	filter := &metadata.Filter{
		AND: []metadata.FieldFilter{
			{Field: "category", Operator: "$eq", Value: "electronics"},
			{Field: "year", Operator: "$eq", Value: "2024"},
		},
	}

	bm := idx.Resolve(filter)
	if bm == nil {
		t.Fatal("expected non-nil bitmap")
	}

	// Should match only node 0
	if !bm.Test(0) {
		t.Error("expected bit 0 set")
	}
	if bm.Test(1) || bm.Test(2) {
		t.Errorf("expected bits 1,2 NOT set, got: %v", bm)
	}
}

func TestMetadataBitmapIndex_OR(t *testing.T) {
	idx := NewMetadataBitmapIndex()

	idx.Set("category", "electronics", 0)
	idx.Set("category", "books", 1)
	idx.Set("category", "clothing", 2)

	// Filter: category=electronics OR category=books
	filter := &metadata.Filter{
		OR: []metadata.Filter{
			{AND: []metadata.FieldFilter{{Field: "category", Operator: "$eq", Value: "electronics"}}},
			{AND: []metadata.FieldFilter{{Field: "category", Operator: "$eq", Value: "books"}}},
		},
	}

	bm := idx.Resolve(filter)
	if bm == nil {
		t.Fatal("expected non-nil bitmap")
	}

	if !bm.Test(0) || !bm.Test(1) {
		t.Error("expected bits 0,1 set")
	}
	if bm.Test(2) {
		t.Error("expected bit 2 NOT set")
	}
}

func TestMetadataBitmapIndex_IN(t *testing.T) {
	idx := NewMetadataBitmapIndex()

	idx.Set("color", "red", 0)
	idx.Set("color", "blue", 1)
	idx.Set("color", "green", 2)
	idx.Set("color", "red", 3)

	// Filter: color IN (red, blue)
	filter := &metadata.Filter{
		AND: []metadata.FieldFilter{
			{Field: "color", Operator: "$in", Values: []any{"red", "blue"}},
		},
	}

	bm := idx.Resolve(filter)
	if bm == nil {
		t.Fatal("expected non-nil bitmap")
	}

	if !bm.Test(0) || !bm.Test(1) || !bm.Test(3) {
		t.Error("expected bits 0,1,3 set")
	}
	if bm.Test(2) {
		t.Error("expected bit 2 NOT set")
	}
}

func TestMetadataBitmapIndex_Remove(t *testing.T) {
	idx := NewMetadataBitmapIndex()

	idx.Set("category", "electronics", 0)
	idx.Set("category", "electronics", 1)
	idx.Remove("category", "electronics", 0)

	filter := &metadata.Filter{
		AND: []metadata.FieldFilter{
			{Field: "category", Operator: "$eq", Value: "electronics"},
		},
	}

	bm := idx.Resolve(filter)
	if bm.Test(0) {
		t.Error("expected bit 0 cleared after remove")
	}
	if !bm.Test(1) {
		t.Error("expected bit 1 still set")
	}
}

func TestMetadataBitmapIndex_EmptyFilter(t *testing.T) {
	idx := NewMetadataBitmapIndex()
	idx.Set("category", "electronics", 0)

	if idx.Resolve(nil) != nil {
		t.Error("expected nil bitmap for nil filter")
	}
}

func TestMetadataBitmapIndex_Size(t *testing.T) {
	idx := NewMetadataBitmapIndex()

	idx.Set("category", "electronics", 0)
	idx.Set("category", "books", 1)
	idx.Set("year", "2024", 0)

	if idx.Size() != 3 {
		t.Errorf("expected size 3, got %d", idx.Size())
	}
	if idx.Cardinality("category") != 2 {
		t.Errorf("expected category cardinality 2, got %d", idx.Cardinality("category"))
	}
}

func TestMetadataBitmapIndex_NumericGT(t *testing.T) {
	idx := NewMetadataBitmapIndex()

	idx.Set("price", 10.0, 0)
	idx.Set("price", 20.0, 1)
	idx.Set("price", 30.0, 2)
	idx.Set("price", 40.0, 3)
	idx.Set("price", 50.0, 4)

	// price > 25
	filter := &metadata.Filter{
		AND: []metadata.FieldFilter{
			{Field: "price", Operator: "$gt", Value: 25.0},
		},
	}

	bm := idx.Resolve(filter)
	if bm == nil {
		t.Fatal("expected non-nil bitmap")
	}

	// Should match nodes 2 (30), 3 (40), 4 (50)
	if !bm.Test(2) || !bm.Test(3) || !bm.Test(4) {
		t.Error("expected bits 2,3,4 set")
	}
	if bm.Test(0) || bm.Test(1) {
		t.Error("expected bits 0,1 NOT set")
	}
}

func TestMetadataBitmapIndex_NumericGTE(t *testing.T) {
	idx := NewMetadataBitmapIndex()

	idx.Set("price", 10.0, 0)
	idx.Set("price", 20.0, 1)
	idx.Set("price", 30.0, 2)

	// price >= 20
	filter := &metadata.Filter{
		AND: []metadata.FieldFilter{
			{Field: "price", Operator: "$gte", Value: 20.0},
		},
	}

	bm := idx.Resolve(filter)
	if !bm.Test(1) || !bm.Test(2) {
		t.Error("expected bits 1,2 set")
	}
	if bm.Test(0) {
		t.Error("expected bit 0 NOT set")
	}
}

func TestMetadataBitmapIndex_NumericLT(t *testing.T) {
	idx := NewMetadataBitmapIndex()

	idx.Set("price", 10.0, 0)
	idx.Set("price", 20.0, 1)
	idx.Set("price", 30.0, 2)

	// price < 20
	filter := &metadata.Filter{
		AND: []metadata.FieldFilter{
			{Field: "price", Operator: "$lt", Value: 20.0},
		},
	}

	bm := idx.Resolve(filter)
	if !bm.Test(0) {
		t.Error("expected bit 0 set")
	}
	if bm.Test(1) || bm.Test(2) {
		t.Error("expected bits 1,2 NOT set")
	}
}

func TestMetadataBitmapIndex_NumericLTE(t *testing.T) {
	idx := NewMetadataBitmapIndex()

	idx.Set("price", 10.0, 0)
	idx.Set("price", 20.0, 1)
	idx.Set("price", 30.0, 2)

	// price <= 20
	filter := &metadata.Filter{
		AND: []metadata.FieldFilter{
			{Field: "price", Operator: "$lte", Value: 20.0},
		},
	}

	bm := idx.Resolve(filter)
	if !bm.Test(0) || !bm.Test(1) {
		t.Error("expected bits 0,1 set")
	}
	if bm.Test(2) {
		t.Error("expected bit 2 NOT set")
	}
}

func TestMetadataBitmapIndex_NumericAndCategory(t *testing.T) {
	idx := NewMetadataBitmapIndex()

	// Node 0: category=electronics, price=10
	idx.Set("category", "electronics", 0)
	idx.Set("price", 10.0, 0)

	// Node 1: category=electronics, price=50
	idx.Set("category", "electronics", 1)
	idx.Set("price", 50.0, 1)

	// Node 2: category=books, price=30
	idx.Set("category", "books", 2)
	idx.Set("price", 30.0, 2)

	// Filter: category=electronics AND price > 20
	filter := &metadata.Filter{
		AND: []metadata.FieldFilter{
			{Field: "category", Operator: "$eq", Value: "electronics"},
			{Field: "price", Operator: "$gt", Value: 20.0},
		},
	}

	bm := idx.Resolve(filter)
	if bm == nil {
		t.Fatal("expected non-nil bitmap")
	}

	// Should match only node 1 (electronics + price=50)
	if !bm.Test(1) {
		t.Error("expected bit 1 set")
	}
	if bm.Test(0) || bm.Test(2) {
		t.Error("expected bits 0,2 NOT set")
	}
}
