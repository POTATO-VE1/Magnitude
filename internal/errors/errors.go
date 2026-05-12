// Package errors defines the canonical error hierarchy for the VectorDB system.
// All errors produced anywhere in the codebase are wrapped in VDBError so that
// API handlers can map them to precise HTTP status codes without brittle string matching.
package errors

import (
	"fmt"
	"net/http"
)

// ErrorCode is a stable numeric code identifying a class of error.
// Codes are exposed in API responses so clients can handle them programmatically.
type ErrorCode int

const (
	// ErrDimensionMismatch is returned when a vector's dimension does not match
	// the index's configured dimension. Maps to HTTP 400.
	ErrDimensionMismatch ErrorCode = iota + 1

	// ErrCollectionNotFound is returned when referencing a collection that does
	// not exist in the metadata store. Maps to HTTP 404.
	ErrCollectionNotFound

	// ErrVectorNotFound is returned when a Delete or point-lookup targets an ID
	// that does not exist in the index. Maps to HTTP 404.
	ErrVectorNotFound

	// ErrCapacityExceeded is returned when an insert would exceed the index's
	// configured maximum capacity. Maps to HTTP 507 (Insufficient Storage).
	ErrCapacityExceeded

	// ErrIndexNotBuilt is returned when a Search is attempted on an index that
	// has not been built yet (e.g., IVF before the first K-Means run). Maps to HTTP 503.
	ErrIndexNotBuilt

	// ErrCorruptedFile is returned when an on-disk file fails its magic-bytes or
	// length-prefix validation during load. Maps to HTTP 500.
	ErrCorruptedFile

	// ErrConcurrentModification is returned when a concurrent write conflicts with
	// an ongoing rebuild or compaction. Maps to HTTP 409.
	ErrConcurrentModification

	// ErrDuplicateID is returned when Insert is called with a pre-existing ID.
	// Some clients may treat this as success (idempotent insert). Maps to HTTP 409.
	ErrDuplicateID

	// ErrRateLimitExceeded is returned when a request exceeds the per-IP or
	// per-key rate limit. Maps to HTTP 429.
	ErrRateLimitExceeded

	// ErrUnauthorized is returned when authentication fails. Never include key
	// material in the error message. Maps to HTTP 401.
	ErrUnauthorized

	// ErrIntegrityFailure is returned when an on-disk checksum (CRC32/SHA256)
	// does not match the stored value at load time. Maps to HTTP 500.
	ErrIntegrityFailure

	// ErrQuotaExceeded is returned when a tenant's vector or request quota is
	// exhausted. Maps to HTTP 402.
	ErrQuotaExceeded

	// ErrTenantNotFound is returned when an API call references a tenant that
	// does not exist in the SysDB. Maps to HTTP 404.
	ErrTenantNotFound
)

// VDBError is the canonical error type for the entire system.
// It wraps an ErrorCode, a human-readable message, and an optional cause
// for error chain traversal with errors.Is / errors.As.
type VDBError struct {
	Code    ErrorCode
	Message string
	Cause   error
}

// Error implements the error interface. Format: "[VDB-N] message".
func (e *VDBError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("[VDB-%d] %s: %v", e.Code, e.Message, e.Cause)
	}
	return fmt.Sprintf("[VDB-%d] %s", e.Code, e.Message)
}

// Unwrap implements the errors.Unwrap protocol, enabling errors.Is and errors.As
// to traverse the error chain.
func (e *VDBError) Unwrap() error {
	return e.Cause
}

// New constructs a VDBError with the given code, message, and optional cause.
// Pass nil for cause if there is no underlying error to wrap.
func New(code ErrorCode, message string, cause error) *VDBError {
	return &VDBError{
		Code:    code,
		Message: message,
		Cause:   cause,
	}
}

// Newf constructs a VDBError with a formatted message and no cause.
func Newf(code ErrorCode, format string, args ...any) *VDBError {
	return &VDBError{
		Code:    code,
		Message: fmt.Sprintf(format, args...),
	}
}

// HTTPStatusCode maps a VDBError's ErrorCode to the appropriate HTTP status code.
// Used by API handlers to produce correct HTTP responses without brittle switch statements
// scattered across handler code.
func (e *VDBError) HTTPStatusCode() int {
	switch e.Code {
	case ErrDimensionMismatch:
		return http.StatusBadRequest // 400
	case ErrCollectionNotFound, ErrVectorNotFound, ErrTenantNotFound:
		return http.StatusNotFound // 404
	case ErrUnauthorized:
		return http.StatusUnauthorized // 401
	case ErrQuotaExceeded:
		return http.StatusPaymentRequired // 402
	case ErrDuplicateID, ErrConcurrentModification:
		return http.StatusConflict // 409
	case ErrRateLimitExceeded:
		return http.StatusTooManyRequests // 429
	case ErrCapacityExceeded:
		return http.StatusInsufficientStorage // 507
	case ErrIndexNotBuilt:
		return http.StatusServiceUnavailable // 503
	case ErrCorruptedFile, ErrIntegrityFailure:
		return http.StatusInternalServerError // 500
	default:
		return http.StatusInternalServerError // 500
	}
}

// Is enables errors.Is() comparison against ErrorCode values.
// Usage: errors.Is(err, &VDBError{Code: ErrDimensionMismatch})
func (e *VDBError) Is(target error) bool {
	t, ok := target.(*VDBError)
	if !ok {
		return false
	}
	return e.Code == t.Code
}

// IsCode is a convenience helper that checks if an error is a VDBError
// with the specified ErrorCode without needing to type-assert manually.
//
//	if vdberrors.IsCode(err, vdberrors.ErrDuplicateID) { ... }
func IsCode(err error, code ErrorCode) bool {
	for err != nil {
		if e, ok := err.(*VDBError); ok && e.Code == code {
			return true
		}
		u, ok := err.(interface{ Unwrap() error })
		if !ok {
			break
		}
		err = u.Unwrap()
	}
	return false
}
