#!/usr/bin/env python3
"""FastEmbed CLI for embedx - reads JSON from stdin, writes JSON to stdout.

Supports:
- Multiple model caching with lazy loading
- Pull (pre-download) models
- List cached models
- Hot-switch model via protocol
- Chroma vector store (collections, add, query, delete)
"""

import os
import sys
import json
import shutil
import traceback
import threading
import time
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from fastembed import TextEmbedding

# Fallback mirror when huggingface.co is inaccessible
HF_FALLBACK_ENDPOINT = "https://hf-mirror.com"


def _is_network_error(exc: Exception) -> bool:
    """Return True if exception looks like a network/download failure."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    network_keywords = (
        "timeout", "connect", "network", "read error", "read timed out",
        "connection refused", "connection reset", "dns", "proxy",
        "host unreachable", "no route to host", "httpcore", "httpx",
        "sslv3", "ssl", "tls", "certificate", "handshake",
    )
    return any(k in msg or k in name for k in network_keywords)


def _try_load_model(model_name: str, endpoint: Optional[str] = None):
    """Load TextEmbedding, optionally using a specific HF endpoint."""
    old_endpoint = os.environ.get("HF_ENDPOINT")
    try:
        if endpoint:
            os.environ["HF_ENDPOINT"] = endpoint
        else:
            os.environ.pop("HF_ENDPOINT", None)
        return TextEmbedding(model_name=model_name)
    finally:
        if old_endpoint is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = old_endpoint


def _load_with_fallback(model_name: str):
    """Load model, falling back to hf-mirror.com on network failure."""
    try:
        return _try_load_model(model_name)
    except Exception as primary_err:
        if not _is_network_error(primary_err):
            raise
        log_warn("Primary HF endpoint failed, trying fallback", endpoint=HF_FALLBACK_ENDPOINT, error=str(primary_err))
        return _try_load_model(model_name, endpoint=HF_FALLBACK_ENDPOINT)

# Use local ./models directory as HuggingFace cache (set by service env or defaults)
# This avoids repeated downloads and ensures offline capability after first load
if "HF_HUB_CACHE" not in os.environ:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["HF_HUB_CACHE"] = os.path.join(_script_dir, "models")
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HUB_CACHE"])

# Default cache directory (FastEmbed default)
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "fastembed"

# Logging utility
def log(level: str, msg: str, **kwargs):
    ts = time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    parts = [f"[{ts}] [{level}] {msg}"]
    for k, v in kwargs.items():
        parts.append(f" {k}={v}")
    print("".join(parts), flush=True, file=sys.stderr)

def log_info(msg: str, **kwargs):
    log("INFO", msg, **kwargs)

def log_warn(msg: str, **kwargs):
    log("WARN", msg, **kwargs)

def log_error(msg: str, **kwargs):
    log("ERROR", msg, **kwargs)

def log_debug(msg: str, **kwargs):
    log("DEBUG", msg, **kwargs)

def log_traceback(exc: Exception, context: str = ""):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log_error(f"EXCEPTION in {context}", exc_type=type(exc).__name__, exc_msg=str(exc), traceback=tb)


# Model cache: name -> TextEmbedding instance
_model_cache: dict[str, TextEmbedding] = {}
_current_model: Optional[str] = None
_cache_lock = threading.Lock()

# Chroma client (persistent)
_chroma_client: Optional[chromadb.PersistentClient] = None
_chroma_lock = threading.Lock()


def get_chroma_client() -> chromadb.PersistentClient:
    """Get or create Chroma client."""
    global _chroma_client
    if _chroma_client is None:
        with _chroma_lock:
            if _chroma_client is None:
                chroma_dir = os.path.expanduser("~/.embedx/chroma")
                os.makedirs(chroma_dir, exist_ok=True)
                _chroma_client = chromadb.PersistentClient(path=chroma_dir)
                log_info("Chroma client initialized", path=chroma_dir)
    return _chroma_client


def chroma_create(name: str, dimension: int = 768, metric: str = "cosine"):
    """Create a Chroma collection."""
    log_info("chroma_create", name=name, dimension=dimension, metric=metric)
    client = get_chroma_client()
    try:
        collection = client.get_or_create_collection(
            name=name,
            metadata={"dimension": dimension, "metric": metric}
        )
        count = collection.count()
        log_info("chroma_create done", name=name, count=count)
        return {"status": "created", "name": name, "count": count}
    except Exception as e:
        log_traceback(e, f"chroma_create {name}")
        return {"status": "error", "error": str(e)}


def chroma_list():
    """List all Chroma collections."""
    log_info("chroma_list")
    client = get_chroma_client()
    try:
        collections = client.list_collections()
        result = [{"name": c.name, "metadata": c.metadata} for c in collections]
        log_info("chroma_list done", count=len(result))
        return {"status": "ok", "collections": result}
    except Exception as e:
        log_traceback(e, "chroma_list")
        return {"status": "error", "error": str(e)}


def chroma_delete(name: str):
    """Delete a Chroma collection."""
    log_info("chroma_delete", name=name)
    client = get_chroma_client()
    try:
        client.delete_collection(name)
        log_info("chroma_delete done", name=name)
        return {"status": "deleted", "name": name}
    except Exception as e:
        log_traceback(e, f"chroma_delete {name}")
        return {"status": "error", "error": str(e)}


def chroma_add(name: str, ids: list[str], embeddings: list[list[float]], metadatas: Optional[list[dict]] = None):
    """Add vectors to a Chroma collection."""
    log_info("chroma_add", name=name, count=len(ids))
    client = get_chroma_client()
    try:
        collection = client.get_or_create_collection(name=name)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas or [{} for _ in ids]
        )
        count = collection.count()
        log_info("chroma_add done", name=name, total_count=count)
        return {"status": "added", "name": name, "count": len(ids), "total": count}
    except Exception as e:
        log_traceback(e, f"chroma_add {name}")
        return {"status": "error", "error": str(e)}


def chroma_query(name: str, query_embeddings: list[list[float]], n_results: int = 10, include: Optional[list[str]] = None):
    """Query similar vectors from a Chroma collection."""
    log_info("chroma_query", name=name, n_results=n_results, query_count=len(query_embeddings))
    client = get_chroma_client()
    try:
        collection = client.get_or_create_collection(name=name)
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=include or ["metadatas", "distances"]
        )
        log_info("chroma_query done", name=name, result_ids=len(results.get("ids", [[]])[0]))
        return {
            "status": "ok",
            "ids": results.get("ids", []),
            "distances": results.get("distances", []),
            "metadatas": results.get("metadatas", []),
            "embeddings": results.get("embeddings", [])
        }
    except Exception as e:
        log_traceback(e, f"chroma_query {name}")
        return {"status": "error", "error": str(e)}


def chroma_delete_vectors(name: str, ids: list[str]):
    """Delete vectors from a Chroma collection."""
    log_info("chroma_delete_vectors", name=name, count=len(ids))
    client = get_chroma_client()
    try:
        collection = client.get_or_create_collection(name=name)
        collection.delete(ids=ids)
        count = collection.count()
        log_info("chroma_delete_vectors done", name=name, remaining=count)
        return {"status": "deleted", "name": name, "count": len(ids), "remaining": count}
    except Exception as e:
        log_traceback(e, f"chroma_delete_vectors {name}")
        return {"status": "error", "error": str(e)}


def chroma_reset(name: str):
    """Reset (clear) a Chroma collection."""
    log_info("chroma_reset", name=name)
    client = get_chroma_client()
    try:
        collection = client.get_or_create_collection(name=name)
        collection.delete(where={})  # Delete all
        log_info("chroma_reset done", name=name)
        return {"status": "reset", "name": name}
    except Exception as e:
        log_traceback(e, f"chroma_reset {name}")
        return {"status": "error", "error": str(e)}


def setup_proxy():
    """Log proxy settings from standard environment variables."""
    http_proxy = os.environ.get("HTTP_PROXY", "") or os.environ.get("http_proxy", "")
    https_proxy = os.environ.get("HTTPS_PROXY", "") or os.environ.get("https_proxy", "")
    no_proxy = os.environ.get("NO_PROXY", "") or os.environ.get("no_proxy", "")
    if http_proxy or https_proxy:
        log_info("Proxy active", HTTP=http_proxy, HTTPS=https_proxy, NO_PROXY=no_proxy)
    else:
        log_info("Proxy: not set (direct connection)")


def embed(texts, model_name: Optional[str] = None):
    """Generate embeddings for a list of texts"""
    global _model_cache, _current_model

    target = model_name or _current_model
    if not target:
        raise RuntimeError("No model loaded. Use 'load' command first.")

    log_debug("embed called", target=target, num_texts=len(texts), total_len=sum(len(t) for t in texts))

    # Load model if not cached
    if target not in _model_cache:
        log_info("Model not cached, loading", model=target)
        try:
            start = time.time()
            with _cache_lock:
                _model_cache[target] = TextEmbedding(model_name=target)
            elapsed = time.time() - start
            log_info("Model loaded", model=target, elapsed_ms=int(elapsed*1000), cached=list(_model_cache.keys()))
        except Exception as e:
            log_traceback(e, f"loading model {target}")
            print(json.dumps({"status": "error", "model": target, "error": str(e)}), flush=True)
            raise

    # Generate embeddings
    try:
        start = time.time()
        embeddings = list(_model_cache[target].embed(texts))
        elapsed = time.time() - start
        result = [[float(x) for x in emb] for emb in embeddings]
        log_debug("Embed done", model=target, num=len(texts), elapsed_ms=int(elapsed*1000), embedding_dim=len(result[0]) if result else 0)
        return result
    except Exception as e:
        log_traceback(e, f"embedding with model {target}")
        raise


def get_cached_models() -> list[str]:
    """Return list of model names that are downloaded in the cache dir."""
    cached = []
    if DEFAULT_CACHE_DIR.exists():
        for item in DEFAULT_CACHE_DIR.iterdir():
            if item.is_dir():
                # Check if it looks like a fastembed cache
                if (item / "config.json").exists() or (item / "model.onnx").exists():
                    cached.append(item.name)
    log_debug("get_cached_models", cached=cached)
    return cached


def pull_model(model_name: str, stream_callback=None):
    """Pull a model (download + verify) and return its dimensions."""
    setup_proxy()
    log_info("pull_model START", model=model_name)
    try:
        start = time.time()
        # Trigger download by creating a temporary embedding instance
        # This will download the model if not cached, with fallback to hf-mirror
        emb = _load_with_fallback(model_name)
        dims = emb.embedding_size
        elapsed = time.time() - start
        log_info("pull_model DONE", model=model_name, dimensions=dims, elapsed_ms=int(elapsed*1000), cached=get_cached_models())
        print(json.dumps({
            "status": "pulled",
            "model": model_name,
            "dimensions": dims,
            "cached": get_cached_models()
        }), flush=True)
        return dims
    except Exception as e:
        log_traceback(e, f"pull_model {model_name}")
        print(json.dumps({"status": "error", "model": model_name, "error": str(e)}), flush=True)
        raise


def load_model(model_name: str):
    """Load a model into the cache (download if needed, then cache)."""
    global _current_model

    log_info("load_model START", model=model_name)
    setup_proxy()

    if model_name not in _model_cache:
        log_info("Model not in cache, loading", model=model_name)
        try:
            start = time.time()
            with _cache_lock:
                _model_cache[model_name] = _load_with_fallback(model_name)
            elapsed = time.time() - start
            log_info("Model loaded into cache", model=model_name, elapsed_ms=int(elapsed*1000))
        except Exception as e:
            log_traceback(e, f"load_model {model_name}")
            print(json.dumps({"status": "error", "model": model_name, "error": str(e)}), flush=True, file=sys.stderr)
            raise

    with _cache_lock:
        _current_model = model_name
        desc = _model_cache[model_name]

    log_info("load_model COMPLETE", model=model_name, dimensions=desc.dim, max_length=desc.max_length, cached=list(_model_cache.keys()))
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

    log_info("unload_model", model=model_name)
    with _cache_lock:
        if model_name in _model_cache:
            del _model_cache[model_name]
            log_info("Model unloaded", model=model_name, cached=list(_model_cache.keys()))
            print(json.dumps({"status": "unloaded", "model": model_name, "cached": list(_model_cache.keys())}), flush=True)
        else:
            log_warn("Model not in cache", model=model_name)
            print(json.dumps({"status": "not_cached", "model": model_name}), flush=True)

        if _current_model == model_name:
            _current_model = None


def handle_command(data: dict) -> dict:
    """Handle a command from Go backend. Returns dict to send as JSON response."""
    cmd = data.get("command", "")
    model = data.get("model", "")
    model_name = data.get("model_name", model)

    log_debug("handle_command", cmd=cmd, model=model, model_name=model_name, data_keys=list(data.keys()))

    try:
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
            log_debug("embed request", num_texts=len(texts), model_name=model_name)
            embeddings = embed(texts, model_name)
            return {"type": "embed_done", "embeddings": embeddings}

        # Chroma commands
        elif cmd == "chroma_create":
            name = data.get("name", "default")
            dimension = data.get("dimension", 768)
            metric = data.get("metric", "cosine")
            return chroma_create(name, dimension, metric)

        elif cmd == "chroma_list":
            return chroma_list()

        elif cmd == "chroma_delete":
            name = data.get("name", "default")
            return chroma_delete(name)

        elif cmd == "chroma_add":
            name = data.get("name", "default")
            ids = data.get("ids", [])
            embeddings = data.get("embeddings", [])
            metadatas = data.get("metadatas")
            return chroma_add(name, ids, embeddings, metadatas)

        elif cmd == "chroma_query":
            name = data.get("name", "default")
            query_embeddings = data.get("query_embeddings", [])
            n_results = data.get("n_results", 10)
            include = data.get("include")
            return chroma_query(name, query_embeddings, n_results, include)

        elif cmd == "chroma_delete":
            name = data.get("name", "default")
            ids = data.get("ids")
            if ids:
                return chroma_delete_vectors(name, ids)
            else:
                return chroma_delete(name)

        elif cmd == "chroma_reset":
            name = data.get("name", "default")
            return chroma_reset(name)

        else:
            log_warn("Unknown command", cmd=cmd)
            return {"type": "error", "error": f"Unknown command: {cmd}"}
    except Exception as e:
        log_traceback(e, f"handle_command [{cmd}]")
        return {"type": "error", "error": str(e)}


def main():
    global _current_model

    log_info("embedx Python starting", pid=os.getpid(), python_version=sys.version)

    # Setup proxy from environment before any network calls
    setup_proxy()

    default_model = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-small-zh-v1.5"
    log_info("Startup: loading default model", default_model=default_model)

    # Load default model on startup (with fallback to hf-mirror)
    try:
        start = time.time()
        _model_cache[default_model] = _load_with_fallback(default_model)
        _current_model = default_model
        desc = _model_cache[default_model]
        elapsed = time.time() - start
        log_info("Startup: default model loaded", model=default_model, dimensions=desc.embedding_size, elapsed_ms=int(elapsed*1000))
        print(json.dumps({
            "event": "ready",
            "model": default_model,
            "dimensions": desc.embedding_size,
            "cached": list(_model_cache.keys())
        }), flush=True)
    except Exception as e:
        log_traceback(e, "startup model load")
        log_error("Startup: FAILED to load default model", model=default_model, error=str(e))
        # Signal error so Go restarts Python instead of marking it alive with a broken model
        print(json.dumps({"event": "error", "model": default_model, "error": str(e)}), flush=True)

    log_info("Entering command loop", stdin_fd=sys.stdin.fileno())

    # Read JSON commands from stdin, write JSON to stdout
    line_count = 0
    for line in sys.stdin:
        line = line.strip()
        line_count += 1
        if not line:
            log_debug("empty line, skipping", line_num=line_count)
            continue

        log_debug("stdin received", line_num=line_count, line_len=len(line), preview=line[:100])

        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            log_error("Invalid JSON received", line_num=line_count, error=str(e), line_preview=line[:200])
            print(json.dumps({"error": f"Invalid JSON: {line[:100]} - {e}"}), flush=True)
            continue

        try:
            result = handle_command(data)
            result_str = json.dumps(result)
            log_debug("stdout wrote", line_num=line_count, result_len=len(result_str), result_preview=result_str[:100])
            print(json.dumps(result), flush=True)
        except Exception as e:
            log_traceback(e, f"command loop line {line_count}")
            print(json.dumps({"error": str(e)}), flush=True)

    log_error("stdin EOF - command loop exited", total_lines=line_count)


if __name__ == "__main__":
    main()
