// Package grpc implements the gRPC server for Magnitude VectorDB.
package grpc

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/POTATO-VE1/Magnitude/internal/collection"
	pb "github.com/POTATO-VE1/Magnitude/internal/grpc/proto"
)

// Server implements the MagnitudeService gRPC server.
type Server struct {
	pb.UnimplementedMagnitudeServiceServer
	manager *collection.Manager
	grpc    *grpc.Server
	addr    string
}

// NewServer creates a new gRPC server.
func NewServer(addr string, mgr *collection.Manager) *Server {
	return &Server{
		manager: mgr,
		addr:    addr,
	}
}

// Start begins serving gRPC requests.
func (s *Server) Start() error {
	lis, err := net.Listen("tcp", s.addr)
	if err != nil {
		return fmt.Errorf("grpc: listen %s: %w", s.addr, err)
	}

	s.grpc = grpc.NewServer()
	pb.RegisterMagnitudeServiceServer(s.grpc, s)

	slog.Info("gRPC server starting", "addr", s.addr)
	return s.grpc.Serve(lis)
}

// Stop gracefully stops the gRPC server.
func (s *Server) Stop() {
	if s.grpc != nil {
		s.grpc.GracefulStop()
	}
}

// ── Health ──────────────────────────────────────────────────────────────────

func (s *Server) Health(ctx context.Context, req *pb.HealthRequest) (*pb.HealthResponse, error) {
	return &pb.HealthResponse{Status: "ok"}, nil
}

// ── Collections ─────────────────────────────────────────────────────────────

func (s *Server) CreateCollection(ctx context.Context, req *pb.CreateCollectionRequest) (*pb.CollectionResponse, error) {
	metric := req.Metric
	if metric == "" {
		metric = "l2"
	}
	indexType := req.IndexType
	if indexType == "" {
		indexType = "hnsw"
	}

	col, err := s.manager.CreateCollectionScoped(req.TenantId, req.DatabaseId, req.Name, int(req.Dimension), metric, indexType)
	if err != nil {
		return nil, toGRPCError(err)
	}

	return &pb.CollectionResponse{
		Id:          col.ID,
		TenantId:    col.TenantID,
		DatabaseId:  col.DatabaseID,
		Name:        col.Name,
		Dimension:   int32(col.Dimension),
		Metric:      col.Metric,
		IndexType:   col.IndexType,
		CreatedAt:   col.CreatedAt,
		VectorCount: int32(col.VectorCount),
	}, nil
}

func (s *Server) ListCollections(ctx context.Context, req *pb.ListCollectionsRequest) (*pb.ListCollectionsResponse, error) {
	cols, err := s.manager.ListCollectionsScoped(req.TenantId, req.DatabaseId)
	if err != nil {
		return nil, toGRPCError(err)
	}

	result := make([]*pb.CollectionResponse, len(cols))
	for i, col := range cols {
		result[i] = &pb.CollectionResponse{
			Id:          col.ID,
			TenantId:    col.TenantID,
			DatabaseId:  col.DatabaseID,
			Name:        col.Name,
			Dimension:   int32(col.Dimension),
			Metric:      col.Metric,
			IndexType:   col.IndexType,
			CreatedAt:   col.CreatedAt,
			VectorCount: int32(col.VectorCount),
		}
	}

	return &pb.ListCollectionsResponse{Collections: result}, nil
}

func (s *Server) DeleteCollection(ctx context.Context, req *pb.DeleteCollectionRequest) (*pb.DeleteResponse, error) {
	if err := s.manager.DeleteCollectionScoped(req.TenantId, req.CollectionId); err != nil {
		return nil, toGRPCError(err)
	}
	return &pb.DeleteResponse{Deleted: req.CollectionId}, nil
}

// ── Vectors ─────────────────────────────────────────────────────────────────

func (s *Server) Insert(ctx context.Context, req *pb.InsertRequest) (*pb.InsertResponse, error) {
	ids := req.Ids
	vectors := make([][]float32, len(req.Vectors))
	for i, v := range req.Vectors {
		vectors[i] = v.Values
	}

	var metadata []map[string]any
	if len(req.Metadata) > 0 {
		metadata = make([]map[string]any, len(req.Metadata))
		for i, m := range req.Metadata {
			var meta map[string]any
			if err := json.Unmarshal(m, &meta); err != nil {
				return nil, status.Errorf(codes.InvalidArgument, "invalid metadata at index %d: %v", i, err)
			}
			metadata[i] = meta
		}
	}

	if err := s.manager.InsertVectors(ctx, req.CollectionId, ids, vectors, metadata); err != nil {
		return nil, toGRPCError(err)
	}

	return &pb.InsertResponse{Inserted: int32(len(ids))}, nil
}

func (s *Server) Search(ctx context.Context, req *pb.SearchRequest) (*pb.SearchResponse, error) {
	k := int(req.K)
	if k <= 0 {
		k = 10
	}
	if k > 10000 {
		k = 10000
	}

	var filter map[string]any
	if len(req.Filter) > 0 {
		if err := json.Unmarshal(req.Filter, &filter); err != nil {
			return nil, status.Errorf(codes.InvalidArgument, "invalid filter: %v", err)
		}
	}

	results, err := s.manager.SearchVectors(ctx, req.CollectionId, req.Query, k, int(req.Nprobe), filter)
	if err != nil {
		return nil, toGRPCError(err)
	}

	pbResults := make([]*pb.SearchResult, len(results))
	for i, r := range results {
		pbResults[i] = &pb.SearchResult{
			Id:       r.ID,
			Distance: r.Distance,
			Score:    r.Score,
		}
	}

	return &pb.SearchResponse{Results: pbResults}, nil
}

func (s *Server) DeleteVectors(ctx context.Context, req *pb.DeleteVectorRequest) (*pb.DeleteResponse, error) {
	for _, id := range req.Ids {
		if err := s.manager.DeleteVector(ctx, req.CollectionId, id); err != nil {
			return nil, toGRPCError(err)
		}
	}
	return &pb.DeleteResponse{Deleted: fmt.Sprintf("%d vectors", len(req.Ids))}, nil
}

// ── Helpers ─────────────────────────────────────────────────────────────────

func toGRPCError(err error) error {
	errMsg := err.Error()
	switch {
	case contains(errMsg, "not found"):
		return status.Error(codes.NotFound, "resource not found")
	case contains(errMsg, "already exists"), contains(errMsg, "conflict"):
		return status.Error(codes.AlreadyExists, "resource already exists")
	case contains(errMsg, "invalid"), contains(errMsg, "dimension"):
		return status.Error(codes.InvalidArgument, "invalid request")
	default:
		return status.Error(codes.Internal, "internal server error")
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsSubstr(s, substr))
}

func containsSubstr(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
