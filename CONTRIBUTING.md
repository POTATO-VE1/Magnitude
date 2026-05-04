# Contributing to Magnitude

Thanks for your interest in contributing. Here's how to get set up.

## Development Setup

### Go Backend

```bash
git clone https://github.com/POTATO-VE1/Magnitude.git
cd Magnitude
go build ./...
go test ./...
go run cmd/server/main.go
```

The server starts on `https://localhost:8443` by default (configurable in `config.yaml`). For development without TLS, remove or comment out the `certFile` and `keyFile` entries — the server will fall back to plain HTTP with a warning.

### Python Client

```bash
cd python-client
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Project Structure

```
cmd/server/      → HTTP server entrypoint
internal/        → Core DB logic (index, storage, metadata, collection, api, distance, ...)
pkg/client/      → Go HTTP client library
python-client/   → CLIP ingest + search CLI
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for a detailed component breakdown.

## Making Changes

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Write tests for new Go code in `_test.go` files alongside the source
4. Run `go test ./...` before pushing
5. Open a pull request with a clear description of what changes and why

## Code Style

- Go: standard `gofmt` formatting. Run `gofmt -w .` before committing.
- Python: PEP 8. Use `black` if available.
- Commit messages: imperative mood, 50-char subject line. E.g. `Add persistent HNSW snapshot support`

## Reporting Issues

Open a GitHub issue with:
- Go version (`go version`)
- OS and architecture
- Steps to reproduce
- Expected vs actual behaviour
