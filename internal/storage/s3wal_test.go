package storage

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestS3WAL_AppendAndRead(t *testing.T) {
	store := NewMemoryStore(0)
	cfg := S3WALConfig{
		Bucket:         "test-bucket",
		Prefix:         "wal3/tenant1/db1/col1",
		FragmentBucket: 4096,
		MaxRetries:     3,
	}

	wal, err := NewS3WAL(store, cfg)
	require.NoError(t, err)

	// Append 3 records
	records := []S3WALRecord{
		{OpType: WALOpInsert, CollectionID: "col1", VectorID: 1, Vector: []float32{1, 2, 3}},
		{OpType: WALOpInsert, CollectionID: "col1", VectorID: 2, Vector: []float32{4, 5, 6}},
		{OpType: WALOpInsert, CollectionID: "col1", VectorID: 3, Vector: []float32{7, 8, 9}},
	}

	lastSeq, err := wal.Append(context.Background(), records)
	require.NoError(t, err)
	assert.Equal(t, uint64(3), lastSeq)
	assert.Equal(t, uint64(3), wal.HeadSeq())
	assert.Equal(t, 1, wal.FragmentCount())

	// Read all records
	read, err := wal.ReadSince(context.Background(), 0)
	require.NoError(t, err)
	assert.Len(t, read, 3)
	assert.Equal(t, uint64(1), read[0].SeqID)
	assert.Equal(t, uint64(3), read[2].SeqID)
}

func TestS3WAL_ReadSince(t *testing.T) {
	store := NewMemoryStore(0)
	cfg := S3WALConfig{
		Bucket:         "test-bucket",
		Prefix:         "wal3/t1/db1/c1",
		FragmentBucket: 4096,
	}

	wal, err := NewS3WAL(store, cfg)
	require.NoError(t, err)

	// Append first batch
	_, err = wal.Append(context.Background(), []S3WALRecord{
		{OpType: WALOpInsert, VectorID: 1},
		{OpType: WALOpInsert, VectorID: 2},
	})
	require.NoError(t, err)

	// Append second batch
	_, err = wal.Append(context.Background(), []S3WALRecord{
		{OpType: WALOpInsert, VectorID: 3},
		{OpType: WALOpDelete, VectorID: 1},
	})
	require.NoError(t, err)

	assert.Equal(t, uint64(4), wal.HeadSeq())
	assert.Equal(t, 2, wal.FragmentCount())

	// Read since seq 2 — should return records 3 and 4
	read, err := wal.ReadSince(context.Background(), 2)
	require.NoError(t, err)
	assert.Len(t, read, 2)
	assert.Equal(t, uint64(3), read[0].SeqID)
	assert.Equal(t, uint64(4), read[1].SeqID)
	assert.Equal(t, WALOpDelete, read[1].OpType)
}

func TestS3WAL_AdvanceCursor(t *testing.T) {
	store := NewMemoryStore(0)
	cfg := S3WALConfig{
		Bucket:         "test-bucket",
		Prefix:         "wal3/t1/db1/c1",
		FragmentBucket: 4096,
	}

	wal, err := NewS3WAL(store, cfg)
	require.NoError(t, err)

	// Write 3 fragments
	for i := 0; i < 3; i++ {
		_, err = wal.Append(context.Background(), []S3WALRecord{
			{OpType: WALOpInsert, VectorID: uint64(i*2 + 1)},
			{OpType: WALOpInsert, VectorID: uint64(i*2 + 2)},
		})
		require.NoError(t, err)
	}

	assert.Equal(t, 3, wal.FragmentCount())
	assert.Equal(t, uint64(6), wal.HeadSeq())

	// Advance cursor past first fragment (seq 1-2)
	require.NoError(t, wal.AdvanceCursor(context.Background(), 3))

	// First fragment should be deleted
	assert.Equal(t, 2, wal.FragmentCount())

	// Only records after seq 3 should remain readable
	read, err := wal.ReadSince(context.Background(), 2)
	require.NoError(t, err)
	assert.Len(t, read, 4) // records 3,4,5,6
}

func TestS3WAL_EmptyAppend(t *testing.T) {
	store := NewMemoryStore(0)
	cfg := S3WALConfig{Bucket: "b", Prefix: "p"}

	wal, err := NewS3WAL(store, cfg)
	require.NoError(t, err)

	seq, err := wal.Append(context.Background(), nil)
	require.NoError(t, err)
	assert.Equal(t, uint64(0), seq)
}

func TestS3WAL_FragmentBucketing(t *testing.T) {
	store := NewMemoryStore(0)
	cfg := S3WALConfig{
		Bucket:         "b",
		Prefix:         "wal3",
		FragmentBucket: 10, // small bucket for testing
	}

	wal, err := NewS3WAL(store, cfg)
	require.NoError(t, err)

	// Write records that span multiple buckets
	for i := 0; i < 5; i++ {
		records := make([]S3WALRecord, 5)
		for j := range records {
			records[j] = S3WALRecord{OpType: WALOpInsert, VectorID: uint64(i*5 + j)}
		}
		_, err = wal.Append(context.Background(), records)
		require.NoError(t, err)
	}

	assert.Equal(t, uint64(25), wal.HeadSeq())

	// Read all and verify continuity
	read, err := wal.ReadSince(context.Background(), 0)
	require.NoError(t, err)
	assert.Len(t, read, 25)

	// Verify sequential seq IDs
	for i, r := range read {
		assert.Equal(t, uint64(i+1), r.SeqID)
	}
}

func TestS3WAL_DeleteOperations(t *testing.T) {
	store := NewMemoryStore(0)
	cfg := S3WALConfig{Bucket: "b", Prefix: "p"}

	wal, err := NewS3WAL(store, cfg)
	require.NoError(t, err)

	// Insert then delete
	_, err = wal.Append(context.Background(), []S3WALRecord{
		{OpType: WALOpInsert, VectorID: 1, Vector: []float32{1, 2, 3}},
		{OpType: WALOpInsert, VectorID: 2, Vector: []float32{4, 5, 6}},
	})
	require.NoError(t, err)

	_, err = wal.Append(context.Background(), []S3WALRecord{
		{OpType: WALOpDelete, VectorID: 1},
	})
	require.NoError(t, err)

	// Read all — should see insert + delete
	read, err := wal.ReadSince(context.Background(), 0)
	require.NoError(t, err)
	assert.Len(t, read, 3)
	assert.Equal(t, WALOpInsert, read[0].OpType)
	assert.Equal(t, WALOpDelete, read[2].OpType)
}
