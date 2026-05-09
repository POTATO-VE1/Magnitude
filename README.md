# Magnitude VectorDB

![Go](https://img.shields.io/badge/Go-1.25+-00ADD8?style=flat&logo=go)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Build](https://img.shields.io/github/actions/workflow/status/POTATO-VE1/Magnitude/go.yml?label=CI)

A lightweight, high-performance vector database built in Go, featuring a custom Python client for CLIP-powered semantic image search.

## Overview

This project implements a barebones yet robust vector search engine from scratch. It consists of two main components:
1.  **Go VectorDB Backend:** A fast, distributed-capable vector database with tenant/database isolation, SQLite-backed metadata, and dense vector embedding support.
2.  **Python Client:** A complete user-layer for batch ingestion and interactive visual search using OpenAI's CLIP model via `sentence-transformers`.

## Why Magnitude?

Most vector database options force a tradeoff you shouldn't have to make:

| Option | Problem |
|---|---|
| Pinecone, Weaviate | Cloud-locked, expensive at scale, data leaves your infra |
| ChromaDB | Python-only, not suitable for Go/polyglot stacks |
| pgvector | Postgres dependency, operational overhead |
| FAISS | No built-in server, no multi-tenancy, C++ ops burden |

**Magnitude** is self-hosted, language-agnostic (Go server + any HTTP client), multi-tenant, and ships a working CLIP semantic search pipeline out of the box. No cloud account required. No per-query billing. Your vectors stay on your machine.

## Features

-   **Multi-Tenancy:** Logical isolation using Tenants, Databases, and Collections with quota enforcement.
-   **High Performance:** Built in Go with pluggable indexing (Flat, IVF, HNSW, SPANN) for rapid similarity search.
-   **SIMD Acceleration:** Optional CGO-based SIMD distance kernels with automatic pure-Go fallback.
-   **RESTful API:** Clean V2 API for managing resources and executing vector queries, including hybrid dense+sparse search.
-   **Semantic Image Search:** Out-of-the-box Python CLI tool that converts text queries into vectors to find matching images in your local dataset.
-   **Interactive UI:** Search results are rendered instantly in a clean, local HTML/CSS lightbox UI.
-   **Observability:** Built-in Prometheus metrics, pprof profiling, and structured JSON logging via `log/slog`.
-   **Production-Ready:** TLS support, API key authentication, rate limiting, graceful shutdown, and config hot-reload via SIGHUP.
-   **Bloom Filters:** Segment-level bloom filters for fast negative lookups, skipping unnecessary disk reads during search.
-   **Crash-Safe Compaction:** Atomic multi-file compaction with action file recovery — no orphaned temp files after crashes.
-   **Priority Task Scheduler:** Background operations (compaction, GC) yield CPU to foreground request handlers.
-   **Tunable Consistency:** Per-operation consistency levels (One, Quorum, All) for distributed read/write operations.
-   **Client-Side Routing:** Cluster-aware Go client with hash ring routing and auto-resync on topology changes.
-   **Gossip Protocol:** UDP-based cluster membership dissemination with deduplication and idempotent event handling.
-   **Failure Detection:** Heartbeat-based node health monitoring with Alive→Suspect→Dead state machine.
-   **Data Migration:** Batch vector transfer between nodes with retry logic and progress tracking.

## Performance

Benchmarked on a standard developer laptop (Apple M2 / AMD Ryzen 7, 16GB RAM). Results may vary by hardware:

| Operation | Dataset Size | Latency |
|---|---|---|
| Vector insert | 1K vectors | ~2ms avg |
| Top-10 query (HNSW ef=64) | 5K vectors | <5ms |
| Top-10 query (HNSW ef=64) | 50K vectors | <12ms |
| Batch ingest (CLIP, bs=16) | 5K images | ~8 min |

> HNSW delivers sub-linear query scaling — doubling the dataset does not double query time.

## Prerequisites & Installation

Before running the project, you need `git`, `Go` (1.25+), and `Python` (3.9+) installed on your system. Run the command for your Operating System below:

**Ubuntu / Debian**
```bash
sudo apt update && sudo apt install git golang python3 python3-venv python3-pip
```

**Fedora**
```bash
sudo dnf install git golang python3 python3-pip
```

**Arch Linux**
```bash
sudo pacman -S git go python python-pip python-virtualenv
```

**macOS (via Homebrew)**
```bash
brew install git go python
```

**Windows (via Winget)**
```powershell
winget install Git.Git GoLang.Go Python.Python.3.12
```

## Quick Start

### 1. Clone the Repository
Download the code to your local machine:
```bash
git clone https://github.com/POTATO-VE1/Magnitude.git
cd Magnitude
```

### 2. Generate TLS Certificates
The server defaults to HTTPS. Generate a self-signed cert for local development:
```bash
mkdir -p certs && openssl req -x509 -newkey rsa:4096 \
  -keyout certs/server.key -out certs/server.crt \
  -days 365 -nodes -subj '/CN=localhost'
```

### 3. Start the Go Backend Server
Start the high-performance Go database server in the background (or in a separate terminal):
```bash
go run cmd/server/main.go
```

### 4. Setup the Python Client
Open a new terminal window, navigate to the client folder, and install the AI dependencies. (Note: The CLIP model weights will be downloaded automatically on the first run).
```bash
cd python-client
python3 -m venv venv

# Activate the virtual environment:
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

pip install -r requirements.txt
```

### 5. Download a Sample Image Dataset
If you don't have a folder of images ready, you can download a 5,000 image dataset (COCO 2017 Validation Set - ~800MB) to really test the database's power:
```bash
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
```

Or, for a quick test, you can download just a few sample images:
```bash
mkdir -p sample_images
cd sample_images
curl -LO https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg
curl -LO https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg
cd ..
```

### 6. Ingest Images
Point the ingest script to your image directory (e.g., `val2017` if you downloaded the 5k dataset). This will convert your images into 512-dimensional embeddings and store them in the VectorDB:
```bash
python ingest.py --dir val2017 --batch-size 16
```

### 7. Semantic Search!
Run the interactive search prompt:
```bash
python search.py
```
Type a query (e.g., "a dog playing" or "a cute cat") and your browser will automatically open with the top visual matches ranked by similarity!

## Architecture

-   **Backend:** Go, SQLite (SysDB/WAL via `modernc.org/sqlite`), Chi Router, Prometheus, pprof
-   **Indexing:** Flat, IVF, HNSW, SPANN — pluggable via config
-   **Distance:** L2, Cosine — with optional SIMD acceleration
-   **Storage:** WAL (SQLite with binary encoding), mmap segments, bloom filters, crash-safe compaction
-   **Distribution:** Gossip protocol, failure detection, tunable consistency, data migration
-   **Scheduling:** Priority-aware task scheduler for background operations
-   **Frontend/Client:** Python 3, Requests, Rich (CLI styling), Sentence-Transformers (CLIP)

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for a detailed system diagram and component breakdown.

## AI-Assisted Development

Magnitude was built with significant AI agent assistance. Architecture decisions, debugging sessions, API design, and documentation were all developed collaboratively with Claude Code and Antigravity.

## Roadmap

- [ ] **Docker Compose** — one-command server deployment
- [ ] **gRPC interface** — alongside REST for high-throughput use cases
- [ ] **Persistent HNSW snapshots** — save/restore index state without re-ingestion
- [ ] **Benchmark CI** — nightly automated perf regression tracking
- [ ] **Web UI** — browser-based collection explorer and search interface
- [ ] **Pluggable embedding providers** — support additional embedding models beyond CLIP for richer cross-modal understanding

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
