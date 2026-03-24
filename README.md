# embedx

A drop-in replacement for Ollama's embedding API, powered by FastEmbed.

## Features

- **Ollama-compatible API** - Uses the same `/api/embeddings` endpoint as Ollama
- **FastEmbed backend** - High-quality Chinese and English embeddings
- **Easy migration** - Replace Ollama's embedding endpoint with embedx
- **Zero configuration** - Works out of the box with sensible defaults

## Architecture

```
HTTP API (embedx :11434)
    └── FastEmbed Python server (127.0.0.1:xxxxx)
            └── BAAI/bge-small-zh-v1.5 model
```

## Quick Start

### 1. Build

```bash
go build -o embedx .
```

### 2. Run

```bash
./embedx
```

### 3. Test

```bash
curl -X POST http://localhost:11434/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-small-zh-v1.5", "prompt": "你好世界"}'
```

## API Endpoints

### POST /api/embeddings

Ollama-compatible embedding endpoint.

**Request:**
```json
{
  "model": "BAAI/bge-small-zh-v1.5",
  "prompt": "your text here"
}
```

**Response:**
```json
{
  "embedding": [0.003, 0.064, -0.045, ...]
}
```

### GET /api/tags

List available models.

### GET /health

Health check endpoint.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDX_PORT` | `11434` | HTTP server port |
| `EMBEDX_MODEL` | `BAAI/bge-small-zh-v1.5` | Default embedding model |

## Supported Models

| Model | Dimensions | Languages | Description |
|-------|------------|-----------|-------------|
| `BAAI/bge-small-zh-v1.5` | 512 | Chinese + English | Default, optimized for Chinese |

## Systemd Service

Create `/etc/systemd/system/embedx.service`:

```ini
[Unit]
Description=embedx - FastEmbed with Ollama-compatible API
After=network.target

[Service]
Type=simple
User=<your-user>
WorkingDirectory=/opt/embedx
ExecStart=/opt/embedx/embedx
Restart=always
RestartSec=5
Environment=EMBEDX_PORT=11434
Environment=EMBEDX_MODEL=BAAI/bge-small-zh-v1.5

[Install]
WantedBy=multi-user.target
```

Install and start:

```bash
sudo cp embedx /opt/embedx/
sudo systemctl daemon-reload
sudo systemctl enable embedx
sudo systemctl start embedx
```

## Migration from Ollama

If you're currently using Ollama for embeddings:

```bash
# Before (Ollama)
OLLAMA_HOST=http://localhost:11434

# After (embedx)
OLLAMA_HOST=http://localhost:11434  # Same URL!
```

No code changes required - the API is identical.

## License

MIT
