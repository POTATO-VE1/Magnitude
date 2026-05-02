# Magnitude VectorDB

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

### 2. Start the Go Backend Server
Start the high-performance Go database server in the background (or in a separate terminal):
```bash
go run cmd/server/main.go
```

### 3. Setup the Python Client
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

### 4. Download a Sample Image Dataset
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

### 5. Ingest Images
Point the ingest script to your image directory (e.g., `val2017` if you downloaded the 5k dataset). This will convert your images into 512-dimensional embeddings and store them in the VectorDB:
```bash
python ingest.py --dir val2017 --batch-size 16
```

### 6. Semantic Search!
Run the interactive search prompt:
```bash
python search.py
```
Type a query (e.g., "a dog playing" or "a cute cat") and your browser will automatically open with the top visual matches ranked by similarity!

## Architecture

-   **Backend:** Go, SQLite (SysDB/WAL), Chi Router
-   **Frontend/Client:** Python 3, Requests, Rich (CLI styling), Sentence-Transformers (CLIP)
