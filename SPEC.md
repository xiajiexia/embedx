# embedx Specification

## Overview

**embedx** is a FastEmbed-powered embedding service with a drop-in Ollama-compatible HTTP API. It uses Go for HTTP and Python subprocess with pipe communication for FastEmbed.

## Architecture

### Components

1. **Go HTTP Server** - Exposes Ollama-compatible API on port 11434
2. **Python Subprocess** - FastEmbed backend, called via stdin/stdout pipes

### Data Flow

```
Client Request
    ↓
Go HTTP Server (:11434)
    ↓ JSON via stdin/stdout pipe
Python Subprocess (FastEmbed)
    ↓
Return embedding to client
```

## Key Design Decisions

- **Go HTTP**: Excellent HTTP performance, Ollama-compatible API
- **Python subprocess via pipe**: Direct communication, no HTTP overhead
- **No separate Python HTTP server**: Simpler architecture, faster

## API Specification

### POST /api/embeddings

Ollama-compatible embedding generation.

**Request Body:**
```json
{
  "model": "string (optional)",
  "prompt": "string (required)"
}
```

**Response:**
```json
{
  "embedding": [float32]
}
```

### GET /api/tags

List available models.

### GET /health

Health check - returns 200 OK.

## Implementation Details

### Communication Protocol

- Go spawns Python subprocess at startup
- Python loads model and signals READY on stdout
- Each embedding request: Go writes JSON to Python stdin, reads JSON from stdout
- Synchronous request/response per call

### Performance

| Component | Performance |
|-----------|-------------|
| Go HTTP | Excellent |
| Pipe communication | Fast (no network stack) |
| FastEmbed inference | CPU-bound |

## File Structure

```
embedx/
├── main.go       # Go HTTP server + subprocess management
├── embed.py      # Python CLI (stdin/stdout)
├── go.mod        # Go module
└── README.md     # Documentation
```

## Status

**Completed:**
- [x] Ollama-compatible `/api/embeddings` endpoint
- [x] Go HTTP server (high performance)
- [x] Python subprocess pipe communication (no HTTP overhead)
- [x] FastEmbed backend integration
- [x] Chinese embedding support
- [x] Configuration via environment variables
- [x] Health check endpoint
