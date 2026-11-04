"""
embeddings.py: meaning-based prompt embeddings for Triage's router.
 
Turns a prompt into 384 numbers with a small sentence-transformer (all-MiniLM-L6-v2). Prompts that mean
similar things get similar vectors, even with different words.
 
Long prompts: the model only reads ~256 tokens at a time. Instead of silently cutting the rest, a long
prompt is split into chunks that fit, each chunk is embedded, and the chunk vectors are averaged
(weighted by chunk length). The batch run reports how many prompts needed this.

Only the user turns are embedded (prompt_text from features.py), the same text the handcrafted features use.
 
The model is loaded once (first use) and reused, so live requests don't pay the load time again.
The first run downloads it (~90 MB); after that it works offline.
 
Usage (batch, for training):
    python -m app.router.embeddings --input data/prompts.jsonl --output data/embeddings.npz
 
Usage (in code, at request time):
    from app.router.embeddings import embed_one
    vec = embed_one("Summarize this email: ...")      # numpy array, shape (384,)
"""

from __future__ import annotations

import argparse 
import json 
import time 
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.router.features import prompt_text

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32
CHUNK_MARGIN = 16   # tokens kept free in each chunk (see chunk_text)

@lru_cache(maxsize=1)
def get_model():
    """Load the sentence-transformer once; later calls return the same object"""
    from sentence_transformers import SentenceTransformer # imported here so importing this file is fast 
    return SentenceTransformer(MODEL_NAME)

def chunk_text(text: str, model) -> list[str]:
    """Split text into pieces the model can read whole. Short text comes back as a single piece."""
    tokenizer = model.tokenizer
    # Room for the model's own start/end tokens, plus a margin: a decoded chunk can re-tokenize a few
    # tokens longer, and the model would silently drop the overflow
    size = model.max_seq_length - 2 - CHUNK_MARGIN
    ids = tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
    if len(ids) <= size:
        return [text]
    return [tokenizer.decode(ids[i:i + size]) for i in range(0, len(ids), size)]

def embed_texts(texts: list[str], model=None) -> tuple[np.ndarray, list[int]]:
    """(vectors of shape (n, dim), number of chunks per text). Each vector has length 1"""
    model = model or get_model()
    pieces, owner = [], []
    for i, text in enumerate(texts):
        for piece in chunk_text(text or "", model):
            pieces.append(piece)
            owner.append(i)
    
    piece_vecs = model.encode(pieces, batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=True,
                              show_progress_bar=len(pieces) > 100)
    weights = np.array([max(len(p), 1) for p in pieces], dtype=np.float64) # longer chunk counts more 
    owner = np.array(owner)
    
    dim = piece_vecs.shape[1]
    out = np.zeros((len(texts), dim), dtype=np.float64)
    np.add.at(out, owner, piece_vecs * weights[:, None]) # sum weighted vectors into their owners
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    out = out / np.where(norms == 0, 1, norms) # back to length 1 after averaging
    chunks = np.bincount(owner, minlength=len(texts)).tolist()
    return out.astype(np.float32), chunks

def embed_one(text: str) -> np.ndarray:
    """One prompt's embedding, for live requests"""
    vecs, _ = embed_texts([text])
    return vecs[0]

# ---------- Batch mode ----------

def main() -> None:
    p = argparse.ArgumentParser(description="Embed every prompt with a sentence-transformer")
    p.add_argument("--input", default="data/prompts.jsonl")
    p.add_argument("--output", default="data/embeddings.npz")
    p.add_argument("--id-field", default="id")
    args = p.parse_args()
    
    records = [json.loads(l) for l in Path(args.input).read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = [str(r[args.id_field]) for r in records]
    texts = [prompt_text(r) for r in records]
    
    start = time.time()
    model = get_model()
    print(f"loaded {MODEL_NAME} in {time.time() - start:.1f}s (reads up to {model.max_seq_length} tokens per chunk)")
    
    start = time.time()
    X, chunks = embed_texts(texts, model)
    print(f"embedded {len(texts)} prompts in {time.time() - start:.1f}s")
    
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ids=np.array(ids), X=X, model_name=np.array(MODEL_NAME),
             max_seq_length=np.array(model.max_seq_length), n_chunks=np.array(chunks))
    
    long = sum(c > 1 for c in chunks)
    print(f"Saved {X.shape[0]} prompts x {X.shape[1]} dims -> {out}")
    print(f"{long} prompts ({long / len(chunks):.1%}) were longer than one chunk and were chunk-averaged"
            + (f"; the longest had {max(chunks)} chunks" if long else ""))


if __name__ == "__main__":
    main()