#!/usr/bin/env python3
"""FastEmbed CLI for embedx - reads JSON from stdin, writes JSON to stdout.

Supports:
- Multiple model caching with lazy loading
- Pull (pre-download) models
- List cached models
- Hot-switch model via protocol
"""

import sys
import json
import os
import shutil
from pathlib import Path
from typing import Optional

from fastembed import TextEmbedding
from fastembed.model_description import ModelSource

# Default cache directory (FastEmbed default)
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "fastembed"

# Model cache: name -> TextEmbedding instance
_model_cache: dict[str, TextEmbedding] = {}
_current_model: Optional[str] = None


def embed(texts, model_name: Optional[str] = None):
    """Generate embeddings for a list of texts"""
    global _model_cache, _current_model

    target = model_name or _current_model
    if not target:
        raise RuntimeError("No model loaded. Use 'load' command first.")

    # Load model if not cached
    if target not in _model_cache:
        print(json.dumps({"status": "loading", "model": target}), flush=True)
        try:
            _model_cache[target] = TextEmbedding(model_name=target)
            print(json.dumps({"status": "loaded", "model": target, "cached": list(_model_cache.keys())}), flush=True)
        except Exception as e:
            print(json.dumps({"status": "error", "model": target, "error": str(e)}), flush=True)
            raise

    embeddings = list(_model_cache[target].embed(texts))
    return [[float(x) for x in emb] for emb in embeddings]


def get_cached_models() -> list[str]:
    """Return list of model names that are downloaded in the cache dir."""
    cached = []
    if DEFAULT_CACHE_DIR.exists():
        for item in DEFAULT_CACHE_DIR.iterdir():
            if item.is_dir():
                # Check if it looks like a fastembed cache
                if (item / "config.json").exists() or (item / "model.onnx").exists():
                    cached.append(item.name)
    return cached


def pull_model(model_name: str, stream_callback=None):
    """Pull a model (download + verify) and return its dimensions."""
    print(json.dumps({"status": "pulling", "model": model_name}), flush=True)
    try:
        # Trigger download by creating a temporary embedding instance
        # This will download the model if not cached
        emb = TextEmbedding(model_name=model_name)
        
        # Get dimensions from the model
        dims = emb.model_description.dim
        
        print(json.dumps({
            "status": "pulled",
            "model": model_name,
            "dimensions": dims,
            "cached": get_cached_models()
        }), flush=True)
        return dims
    except Exception as e:
        print(json.dumps({"status": "error", "model": model_name, "error": str(e)}), flush=True)
        raise


def load_model(model_name: str):
    """Load a model into the cache (download if needed, then cache)."""
    global _current_model

    if model_name not in _model_cache:
        print(json.dumps({"status": "loading", "model": model_name}), flush=True, file=sys.stderr)
        _model_cache[model_name] = TextEmbedding(model_name=model_name)
        print(json.dumps({"status": "loaded", "model": model_name}), flush=True, file=sys.stderr)

    _current_model = model_name
    desc = _model_cache[model_name].model_description
    print(json.dumps({
        "model": model_name,
        "dimensions": desc.dim,
        "max_length": desc.max_length,
        "cached": list(_model_cache.keys())
    }), flush=True)
    return desc


def unload_model(model_name: str):
    """Remove a model from cache."""
    global _current_model

    if model_name in _model_cache:
        del _model_cache[model_name]
        print(json.dumps({"status": "unloaded", "model": model_name, "cached": list(_model_cache.keys())}), flush=True)
    else:
        print(json.dumps({"status": "not_cached", "model": model_name}), flush=True)

    if _current_model == model_name:
        _current_model = None


def handle_command(data: dict) -> dict:
    """Handle a command from Go backend. Returns dict to send as JSON response."""
    cmd = data.get("command", "")
    model = data.get("model", "")
    model_name = data.get("model_name", model)

    if cmd == "pull":
        dims = pull_model(model_name)
        return {"type": "pull_done", "model": model_name, "dimensions": dims}

    elif cmd == "load":
        desc = load_model(model_name)
        return {"type": "model_loaded", "model": model_name, "dimensions": desc.dim}

    elif cmd == "unload":
        unload_model(model_name)
        return {"type": "model_unloaded", "model": model_name}

    elif cmd == "list_cached":
        cached = get_cached_models()
        return {"type": "cached_list", "cached": cached}

    elif cmd == "embed":
        texts = data.get("texts", [])
        if isinstance(texts, str):
            texts = [texts]
        embeddings = embed(texts, model_name)
        return {"type": "embed_done", "embeddings": embeddings}

    else:
        return {"type": "error", "error": f"Unknown command: {cmd}"}


def main():
    global _current_model

    default_model = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-small-zh-v1.5"
    
    # Load default model on startup
    print(f"Loading default model: {default_model}", file=sys.stderr, flush=True)
    try:
        _model_cache[default_model] = TextEmbedding(model_name=default_model)
        _current_model = default_model
        desc = _model_cache[default_model].model_description
        print(json.dumps({
            "event": "ready",
            "model": default_model,
            "dimensions": desc.dim,
            "max_length": desc.max_length,
            "cached": list(_model_cache.keys())
        }), flush=True)
    except Exception as e:
        print(f"Failed to load default model: {e}", file=sys.stderr, flush=True)
        # Still signal ready so Go server can start
        print(json.dumps({"event": "ready", "error": str(e)}), flush=True)

    # Read JSON commands from stdin, write JSON to stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"error": f"Invalid JSON: {line}"}), flush=True)
            continue

        try:
            result = handle_command(data)
            print(json.dumps(result), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
