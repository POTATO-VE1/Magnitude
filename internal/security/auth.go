// Package security — API key authentication.
//
// Authentication model:
//  1. Client sends: Authorization: Bearer <raw-api-key>
//  2. Server computes: SHA-256(raw-api-key) → hex string
//  3. Server checks: hex string ∈ config.Auth.KeyHashes
//  4. Match → authenticated; no match → 401 Unauthorized
//
// Raw API keys are NEVER stored — only their SHA-256 hashes.
// This means a leaked config file does not compromise keys.
package security

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"net/http"
	"strings"

	"github.com/veda/vectordb/internal/metadata"
)

// Authenticator validates API keys using SHA-256 hash comparison.
type Authenticator struct {
	// keyHashes is the set of valid key hashes (hex-encoded).
	// Using a map for O(1) lookup.
	keyHashes map[string]struct{}
	enabled   bool
}

// NewAuthenticator creates an authenticator from a list of SHA-256 key hashes.
// If keyHashes is empty, authentication is disabled (all requests pass).
func NewAuthenticator(keyHashes []string) *Authenticator {
	hashes := make(map[string]struct{}, len(keyHashes))
	for _, h := range keyHashes {
		hashes[strings.ToLower(h)] = struct{}{}
	}
	return &Authenticator{
		keyHashes: hashes,
		enabled:   len(hashes) > 0,
	}
}

// Authenticate validates the Authorization header of an HTTP request.
// Returns true if the request is authenticated.
func (a *Authenticator) Authenticate(r *http.Request) bool {
	if !a.enabled {
		return true // auth disabled
	}

	token := extractBearerToken(r)
	if token == "" {
		return false
	}

	// Compute SHA-256 of the raw key
	hash := sha256.Sum256([]byte(token))
	hexHash := hex.EncodeToString(hash[:])

	// Constant-time comparison against all valid hashes to prevent timing attacks.
	// We iterate all hashes even after finding a match to avoid leaking
	// which hash matched (or didn't) via timing.
	matched := false
	for validHash := range a.keyHashes {
		if subtle.ConstantTimeCompare([]byte(hexHash), []byte(validHash)) == 1 {
			matched = true
		}
	}

	return matched
}

// extractBearerToken extracts the token from "Authorization: Bearer <token>".
func extractBearerToken(r *http.Request) string {
	auth := r.Header.Get("Authorization")
	if auth == "" {
		return ""
	}
	parts := strings.SplitN(auth, " ", 2)
	if len(parts) != 2 || !strings.EqualFold(parts[0], "bearer") {
		return ""
	}
	return strings.TrimSpace(parts[1])
}

// IsEnabled returns whether authentication is active.
func (a *Authenticator) IsEnabled() bool {
	return a.enabled
}

// ── Multi-Tenancy Auth ──────────────────────────────────────────────────────

type contextKey string

const (
	ctxTenantID contextKey = "tenant_id"
	ctxCallerID contextKey = "caller_id" // key hash prefix for audit
	ctxKeyRole  contextKey = "key_role"  // "data" or "admin"
)

// TenantAuthMiddleware resolves the API key to a tenant and role using SysDB.
func TenantAuthMiddleware(sysdb *metadata.SysDB, enabled bool) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if !enabled || r.URL.Path == "/v1/health" || r.URL.Path == "/health" {
				next.ServeHTTP(w, r)
				return
			}

			auth := r.Header.Get("Authorization")
			token, ok := strings.CutPrefix(auth, "Bearer ")
			if !ok || token == "" {
				http.Error(w, `{"error":"missing Authorization header"}`, http.StatusUnauthorized)
				return
			}

			h := sha256.Sum256([]byte(token))
			keyHash := hex.EncodeToString(h[:])

			tenantID, role, err := sysdb.ResolveAPIKey(keyHash)
			if err != nil {
				http.Error(w, `{"error":"unknown API key"}`, http.StatusUnauthorized)
				return
			}

			// Embed TenantID + role into request context
			ctx := r.Context()
			ctx = context.WithValue(ctx, ctxTenantID, tenantID)
			ctx = context.WithValue(ctx, ctxCallerID, keyHash[:8]) // prefix for audit
			ctx = context.WithValue(ctx, ctxKeyRole, role)
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

// TenantID extracts the authenticated tenant ID from the context.
func TenantID(ctx context.Context) string {
	val := ctx.Value(ctxTenantID)
	if val == nil {
		return ""
	}
	return val.(string)
}

// KeyRole extracts the authenticated key role from the context.
func KeyRole(ctx context.Context) string {
	val := ctx.Value(ctxKeyRole)
	if val == nil {
		return ""
	}
	return val.(string)
}
