#!/usr/bin/env python3
"""FastEmbed CLI for embedx - reads JSON from stdin, writes JSON to stdout"""

import sys
import json
from fastembed import TextEmbedding

model = None

def embed(texts):
    """Generate embeddings for a list of texts"""
    global model
    if model is None:
        raise RuntimeError("Model not initialized")
    
    embeddings = list(model.embed(texts))
    # Convert numpy arrays to lists for JSON serialization
    return [[float(x) for x in emb] for emb in embeddings]

def main():
    global model
    
    # Load model from command line argument
    model_name = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-small-zh-v1.5"
    print(f"Loading model: {model_name}", file=sys.stderr, flush=True)
    model = TextEmbedding(model_name=model_name)
    # Signal ready on stdout (Go reads from stdout)
    print("READY", flush=True)
    
    # Read JSON from stdin, write JSON to stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            data = json.loads(line)
            texts = data.get("texts", [])
            if isinstance(texts, str):
                texts = [texts]
            
            embeddings = embed(texts)
            result = {"embeddings": embeddings}
            print(json.dumps(result), flush=True)
        except Exception as e:
            error = {"error": str(e)}
            print(json.dumps(error), flush=True)

if __name__ == "__main__":
    main()
