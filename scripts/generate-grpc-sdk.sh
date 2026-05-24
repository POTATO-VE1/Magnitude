#!/bin/bash
# generate-grpc-sdk.sh — Generate gRPC client SDKs from protobuf definitions.
# Usage: ./scripts/generate-grpc-sdk.sh [language]
#   language: python, typescript, or all (default: all)

set -euo pipefail

PROTO_DIR="internal/grpc/proto"
PROTO_FILE="$PROTO_DIR/magnitude.proto"
LANG="${1:-all}"

generate_python() {
    echo "Generating Python gRPC client..."
    mkdir -p sdks/python-grpc/magnitude_proto

    python3 -m grpc_tools.protoc \
        -I"$PROTO_DIR" \
        --python_out=sdks/python-grpc/magnitude_proto \
        --grpc_python_out=sdks/python-grpc/magnitude_proto \
        "$PROTO_FILE"

    # Create __init__.py
    touch sdks/python-grpc/magnitude_proto/__init__.py

    # Create wrapper client
    cat > sdks/python-grpc/magnitude_proto/client.py << 'PYEOF'
"""Auto-generated gRPC client wrapper for Magnitude."""

import grpc
from magnitude_proto import magnitude_pb2 as pb
from magnitude_proto import magnitude_pb2_grpc as pb_grpc


class MagnitudeGrpcClient:
    """gRPC client for Magnitude vector database."""

    def __init__(self, host: str = "localhost", port: int = 9090):
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = pb_grpc.MagnitudeServiceStub(self.channel)

    def close(self):
        self.channel.close()

    def health(self):
        return self.stub.Health(pb.HealthRequest())

    def create_collection(self, tenant_id, database_id, name, dimension, metric="l2", index_type="hnsw"):
        req = pb.CreateCollectionRequest(
            tenant_id=tenant_id,
            database_id=database_id,
            name=name,
            dimension=dimension,
            metric=metric,
            index_type=index_type,
        )
        return self.stub.CreateCollection(req)

    def list_collections(self, tenant_id, database_id):
        req = pb.ListCollectionsRequest(tenant_id=tenant_id, database_id=database_id)
        return self.stub.ListCollections(req)

    def delete_collection(self, tenant_id, database_id, collection_id):
        req = pb.DeleteCollectionRequest(
            tenant_id=tenant_id, database_id=database_id, collection_id=collection_id
        )
        return self.stub.DeleteCollection(req)

    def insert(self, tenant_id, database_id, collection_id, ids, vectors, metadata=None):
        float_vectors = [pb.FloatVector(values=v) for v in vectors]
        meta_bytes = []
        if metadata:
            import json
            meta_bytes = [json.dumps(m).encode() for m in metadata]
        req = pb.InsertRequest(
            tenant_id=tenant_id,
            database_id=database_id,
            collection_id=collection_id,
            ids=ids,
            vectors=float_vectors,
            metadata=meta_bytes,
        )
        return self.stub.Insert(req)

    def search(self, tenant_id, database_id, collection_id, query, k=10, nprobe=0, filter_dict=None):
        import json
        filter_bytes = json.dumps(filter_dict).encode() if filter_dict else b""
        req = pb.SearchRequest(
            tenant_id=tenant_id,
            database_id=database_id,
            collection_id=collection_id,
            query=query,
            k=k,
            nprobe=nprobe,
            filter=filter_bytes,
        )
        return self.stub.Search(req)

    def delete_vectors(self, tenant_id, database_id, collection_id, ids):
        req = pb.DeleteVectorRequest(
            tenant_id=tenant_id,
            database_id=database_id,
            collection_id=collection_id,
            ids=ids,
        )
        return self.stub.DeleteVectors(req)
PYEOF

    echo "Done: sdks/python-grpc/"
}

generate_typescript() {
    echo "Generating TypeScript gRPC client..."
    mkdir -p sdks/ts-grpc/src/generated

    npx grpc_tools_node_protoc \
        --plugin=protoc-gen-grpc-ts=$(which grpc_tools_node_protoc_plugin_ts 2>/dev/null || echo "node_modules/.bin/grpc_tools_node_protoc_plugin_ts") \
        --js_out=import_style=commonjs,binary:sdks/ts-grpc/src/generated \
        --grpc-ts_out=sdks/ts-grpc/src/generated \
        --proto_path="$PROTO_DIR" \
        "$PROTO_FILE" 2>/dev/null || {
        echo "Note: TypeScript gRPC plugin not installed. Install with:"
        echo "  npm install -g grpc_tools_node_protoc_plugin_ts"
        echo "Generating JS-only fallback..."
        npx grpc_tools_node_protoc \
            --js_out=import_style=commonjs,binary:sdks/ts-grpc/src/generated \
            --grpc_out=grpc_js:sdks/ts-grpc/src/generated \
            --proto_path="$PROTO_DIR" \
            "$PROTO_FILE"
    }

    echo "Done: sdks/ts-grpc/"
}

case $LANG in
    python)
        generate_python
        ;;
    typescript)
        generate_typescript
        ;;
    all)
        generate_python
        generate_typescript
        ;;
    *)
        echo "Unknown language: $LANG"
        echo "Available: python, typescript, all"
        exit 1
        ;;
esac

echo ""
echo "Generated gRPC clients from $PROTO_FILE"
echo "Update the proto file and re-run this script to regenerate."
