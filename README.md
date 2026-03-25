# embedx

A drop-in replacement for Ollama's embedding API, powered by FastEmbed.

## Features

- **Ollama-compatible API** - Uses the same `/api/embeddings` endpoint as Ollama
- **Model hot-switching** - Switch models on-the-fly via `model` parameter
- **Model pull** - Download and cache models via `/api/pull` (Ollama-compatible)
- **Multiple model caching** - Keep multiple models loaded simultaneously
- **FastEmbed backend** - High-quality Chinese and English embeddings
- **Easy migration** - Replace Ollama's embedding endpoint with embedx
- **Zero configuration** - Works out of the box with sensible defaults

## Architecture

```
HTTP API (embedx :11434)
    └── FastEmbed Python server (127.0.0.0)
            └── Cached models in ~/.cache/fastembed
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
# Generate embedding
curl -X POST http://localhost:11434/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-small-zh-v1.5", "prompt": "你好世界"}'

# Pull a model
curl -X POST http://localhost:11434/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "BAAI/bge-base-en-v1.5"}'

# List cached models
curl http://localhost:11434/api/tags
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/embeddings` | POST | Generate embedding |
| `/api/pull` | POST | Download/cache a model |
| `/api/create` | POST | Load model into memory |
| `/api/show` | POST | Get model info |
| `/api/tags` | GET | List cached models |
| `/health` | GET | Health check |

### Switch model via `model` parameter

```bash
curl -X POST http://localhost:11434/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-base-en-v1.5", "prompt": "hello world"}'
```

### Pull with SSE streaming

```bash
curl -X POST http://localhost:11434/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "BAAI/bge-base-en-v1.5", "stream": true}'
```



## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDX_PORT` | `11434` | HTTP server port |
| `EMBEDX_MODEL` | `BAAI/bge-small-zh-v1.5` | Default embedding model |

## Supported Models

All [FastEmbed-supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/).

Popular choices:

| Model | Dimensions | Languages |
|-------|------------|-----------|
| `BAAI/bge-small-zh-v1.5` | 512 | Chinese + English |
| `BAAI/bge-base-en-v1.5` | 768 | English |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | English |
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | English |

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
sudo cp embed.py /opt/embedx/
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
