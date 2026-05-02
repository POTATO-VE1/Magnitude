// Package api — Request context keys for multi-tenancy.
//
// After the TenantResolver middleware runs, every downstream handler can
// extract the tenant and database IDs from the request context using the
// helper functions below. This avoids passing tenant info as explicit
// parameters through the entire call chain.
package api

import (
	"context"
)

type contextKey string

const (
	ctxTenantID   contextKey = "tenant_id"
	ctxDatabaseID contextKey = "database_id"
)

// WithTenant attaches tenant and database IDs to the context.
func WithTenant(ctx context.Context, tenantID, databaseID string) context.Context {
	ctx = context.WithValue(ctx, ctxTenantID, tenantID)
	ctx = context.WithValue(ctx, ctxDatabaseID, databaseID)
	return ctx
}

// TenantID extracts the tenant ID from the request context.
// Returns "" if not set (e.g. v1 API routes).
func TenantID(ctx context.Context) string {
	v, _ := ctx.Value(ctxTenantID).(string)
	return v
}

// DatabaseID extracts the database ID from the request context.
func DatabaseID(ctx context.Context) string {
	v, _ := ctx.Value(ctxDatabaseID).(string)
	return v
}
