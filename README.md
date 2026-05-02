# VectorDB

A lightweight, high-performance vector database built in Go, featuring a custom Python client for CLIP-powered semantic image search.

## Overview

This project implements a barebones yet robust vector search engine from scratch. It consists of two main components:
1.  **Go VectorDB Backend:** A fast, distributed-capable vector database with tenant/database isolation, SQLite-backed metadata, and dense vector embedding support.
2.  **Python Client:** A complete user-layer for batch ingestion and interactive visual search using OpenAI's CLIP model via `sentence-transformers`.

## Features

-   **Multi-Tenancy:** Logical isolation using Tenants, Databases, and Collections.
-   **High Performance:** Built in Go with HNSW indexing for rapid similarity search.
-   **RESTful API:** Clean V2 API for managing resources and executing vector queries.
-   **Semantic Image Search:** Out-of-the-box Python CLI tool that converts text queries into vectors to find matching images in your local dataset.
-   **Interactive UI:** Search results are rendered instantly in a clean, local HTML/CSS lightbox UI.

## Quick Start

### 1. Start the Go Server
Ensure you have Go installed (1.25+), then start the backend server:
```bash
go run cmd/server/main.go
```

### 2. Setup the Python Client
In a new terminal, set up your virtual environment and install dependencies:
```bash
cd python-client
python -m venv venv
source venv/bin/activate  # or activate.fish if using fish shell
pip install -r requirements.txt
```

### 3. Ingest Images
Point the ingest script to a directory of images to generate and store their embeddings in the database:
```bash
python ingest.py --dir path/to/your/images --batch-size 16
```

### 4. Search
Run the interactive REPL to search your database using natural language:
```bash
python search.py
```
Type a query (e.g., "a dog playing in snow") and your browser will automatically open with the top visual matches!

## Architecture

-   **Backend:** Go, SQLite (SysDB/WAL), Chi Router
-   **Frontend/Client:** Python 3, Requests, Rich (CLI styling), Sentence-Transformers (CLIP)
