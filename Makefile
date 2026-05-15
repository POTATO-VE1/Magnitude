.PHONY: run build test clean docker

# Default: run the server
run:
	go run cmd/server/main.go

# Build the binary
build:
	CGO_ENABLED=0 go build -ldflags="-s -w" -o magnitude ./cmd/server/main.go

# Run all tests
test:
	go test ./...

# Build and run Docker container
docker:
	docker compose up --build

# Remove build artifacts
clean:
	rm -f magnitude
	rm -rf data/
