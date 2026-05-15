package gossip

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"sort"
)

// canonicalMarshal produces deterministic JSON by recursively sorting
// map keys before encoding. This ensures all nodes generate identical
// signatures for logically identical messages regardless of Go map
// iteration order.
func canonicalMarshal(v interface{}) ([]byte, error) {
	return json.Marshal(sortValue(v))
}

// sortValue recursively sorts map keys in a value to produce deterministic
// JSON output.
func sortValue(v interface{}) interface{} {
	switch tv := v.(type) {
	case map[string]interface{}:
		keys := make([]string, 0, len(tv))
		for k := range tv {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		sorted := make(map[string]interface{}, len(tv))
		for _, k := range keys {
			sorted[k] = sortValue(tv[k])
		}
		return sorted
	case []interface{}:
		for i, elem := range tv {
			tv[i] = sortValue(elem)
		}
		return tv
	default:
		return v
	}
}

// SignMessage takes a gossip message and a secret key, marshals the message to JSON
// using canonical (sorted-key) encoding, and appends a 32-byte HMAC-SHA256 signature.
// If the secret is empty, it just returns the canonical JSON.
func SignMessage(msg *Message, secret string) ([]byte, error) {
	data, err := canonicalMarshal(msg)
	if err != nil {
		return nil, err
	}
	if secret == "" {
		return data, nil
	}

	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(data)
	signature := mac.Sum(nil)

	// Append signature to data
	return append(data, signature...), nil
}

// VerifyAndUnmarshal takes raw bytes and a secret key, verifies the HMAC-SHA256 signature
// (if a secret is configured), and unmarshals the JSON into a Message.
func VerifyAndUnmarshal(data []byte, secret string) (*Message, error) {
	if secret == "" {
		var msg Message
		if err := json.Unmarshal(data, &msg); err != nil {
			return nil, err
		}
		return &msg, nil
	}

	// Signature is exactly 32 bytes (SHA-256)
	if len(data) < 32 {
		return nil, fmt.Errorf("message too short to contain signature")
	}

	payloadLen := len(data) - 32
	payload := data[:payloadLen]
	signature := data[payloadLen:]

	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(payload)
	expectedMAC := mac.Sum(nil)

	if !hmac.Equal(signature, expectedMAC) {
		return nil, fmt.Errorf("invalid HMAC signature")
	}

	var msg Message
	if err := json.Unmarshal(payload, &msg); err != nil {
		return nil, err
	}
	return &msg, nil
}
