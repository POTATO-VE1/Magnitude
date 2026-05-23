.PHONY: run build test clean docker image-search-setup

# ── Vector Database ──────────────────────────────────────────────────────────

# Run the server
run:
	go run cmd/server/main.go

# Build the binary
build:
	CGO_ENABLED=0 go build -ldflags="-s -w" -o magnitude ./cmd/server/main.go

# Build with CGO (AVX2 SIMD support)
build-simd:
	go build -ldflags="-s -w" -o magnitude ./cmd/server/main.go

# Run all tests
test:
	CGO_CFLAGS_ALLOW="-mavx2|-mfma|-O3" go test ./...

# Build and run Docker container
docker:
	docker compose up --build

# Remove build artifacts
clean:
	rm -f magnitude
	rm -rf data/

# ── Image Search (Optional) ──────────────────────────────────────────────────

# Set up Python environment for image search
image-search-setup:
	cd python-client && python3 -m venv .venv && \
	.venv/bin/pip install -c constraints-cpu.txt -e ".[image-search]"

# Ingest images (requires server running + Python env)
image-ingest:
	cd python-client && .venv/bin/magnitude-ingest --dir ./images --host http://localhost:8080

# Start image search UI
image-ui:
	cd python-client && .venv/bin/magnitude-ui
