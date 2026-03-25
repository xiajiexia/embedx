# embedx Specification

## Overview

**embedx** is a FastEmbed-powered embedding service with a drop-in Ollama-compatible HTTP API. It uses Go for HTTP and Python subprocess with pipe communication for FastEmbed.

## Architecture

### Components

1. **Go HTTP Server** - Exposes Ollama-compatible API on port 11434
2. **Python Subprocess** - FastEmbed backend, called via stdin/stdout pipes with JSON protocol

### Data Flow

```
Client Request (Ollama-compatible API)
    ↓
Go HTTP Server (:11434)
    ↓ JSON via stdin/stdout pipe
Python Subprocess (FastEmbed)
    ↓
Return embedding / status to client
```

## API Specification

### POST /api/embeddings

Ollama-compatible embedding generation.

**Request:**
```json
{
  "model": "BAAI/bge-small-zh-v1.5",   // optional, defaults to EMBEDX_MODEL
  "prompt": "your text here"
}
```

**Response:**
```json
{
  "embedding": [0.003, 0.064, -0.045, ...]
}
```

### POST /api/pull

Ollama-compatible model download.

**Request:**
```json
{
  "name": "BAAI/bge-base-zh-v1.5",
  "stream": false
}
```

**Response (non-streaming):**
```json
{
  "status": "success",
  "model": "BAAI/bge-base-zh-v1.5",
  "dimensions": 768
}
```

**Response (streaming, SSE):**
```
event: status
data: {"status":"pulling","model":"BAAI/bge-base-zh-v1.5"}

event: done
data: {"status":"success","model":"BAAI/bge-base-zh-v1.5"}
```

### POST /api/create

Load a model into memory (does not persist, use `/api/pull` for that).

**Request:**
```json
{"name": "BAAI/bge-base-zh-v1.5"}
```

**Response:**
```json
{"status": "success", "model": "BAAI/bge-base-zh-v1.5", "dimensions": 768}
```

### POST /api/show

Get model information after loading.

**Request:**
```json
{"name": "BAAI/bge-base-zh-v1.5"}
```

### GET /api/tags

List available (cached) models.

**Response:**
```json
{
  "models": [
    {"name": "BAAI/bge-small-zh-v1.5", "model": "BAAI/bge-small-zh-v1.5", "size": 0}
  ]
}
```

### GET /health

Health check - returns 200 OK.

## Python Protocol

Go communicates with Python via JSON messages on stdin/stdout.

### Commands (Go → Python)

```json
{"command": "embed", "model_name": "...", "texts": ["..."]}
{"command": "load", "model_name": "..."}
{"command": "pull", "model_name": "..."}
{"command": "unload", "model_name": "..."}
{"command": "list_cached"}
```

### Responses (Python → Go)

```json
{"type": "embed_done", "embeddings": [[0.1, 0.2, ...]]}
{"type": "model_loaded", "model": "...", "dimensions": 512, "cached": ["..."]}
{"type": "pull_done", "model": "...", "dimensions": 512}
{"type": "error", "error": "..."}
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDX_PORT` | `11434` | HTTP server port |
| `EMBEDX_MODEL` | `BAAI/bge-small-zh-v1.5` | Default embedding model |

## Supported Models

All FastEmbed-supported models. Popular choices:

| Model | Dimensions | Languages |
|-------|------------|-----------|
| `BAAI/bge-small-zh-v1.5` | 512 | Chinese + English |
| `BAAI/bge-base-en-v1.5` | 768 | English |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | English (multimodal) |
| `jinaai/jina-embeddings-v2-base-en` | 768 | English |
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | English |

## File Structure

```
embedx/
├── main.go       # Go HTTP server + subprocess management
├── embed.py      # Python CLI (stdin/stdout protocol)
├── go.mod        # Go module
├── Makefile      # Build commands
├── README.md     # User documentation
└── SPEC.md       # This file
```
