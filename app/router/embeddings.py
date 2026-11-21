"""
embeddings.py: meaning-based prompt embeddings for Triage's router.
 
Turns a prompt into 384 numbers with a small sentence-transformer (all-MiniLM-L6-v2). Prompts that mean
similar things get similar vectors, even with different words.
 
Long prompts: the model only reads ~256 tokens at a time. Instead of silently cutting the rest, a long
prompt is split into chunks that fit, each chunk is embedded, and the chunk vectors are averaged
(weighted by chunk length). The batch run reports how many prompts needed this.

Only the user turns are embedded (prompt_text from features.py), the same text the handcrafted features use.
 
The model is loaded once (first use) and reused, so live requests don't pay the load time again.
The first run downloads it (~90 MB); after that it works offline. A deployed server never downloads it: the build
runs `python -m scripts.fetch_embedding_model`, and the server starts with HF_HUB_OFFLINE=1.

The model runs on plain `transformers` (tokenizer -> transformer -> mean pooling -> L2 normalisation), which is
exactly the all-MiniLM-L6-v2 sentence-transformers pipeline (modules.json: Transformer, Pooling(mean), Normalize).
It avoids importing sentence-transformers, which pulls in scikit-learn, scipy, pandas and datasets: about 200 MB
of memory the server doesn't need. tests/test_embedding_parity.py checks it against sentence-transformers itself.
 
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
# The exact revision the classifier was evaluated with; the build downloads this one (scripts/fetch_embedding_model)
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MAX_SEQ_LENGTH = 256    # the model's sentence_bert_config.json: longer input is cut, so chunk_text splits first
BATCH_SIZE = 32         # batch (training) runs
LIVE_BATCH_SIZE = 4     # live requests: a long prompt's chunks go through a few at a time, to keep memory low
CHUNK_MARGIN = 16   # tokens kept free in each chunk (see chunk_text)
TORCH_THREADS_ENV = "TRIAGE_TORCH_THREADS"


class EmbeddingModelMissing(RuntimeError):
    """The model files aren't in the Hugging Face cache and downloading is off (a deployed server)"""


class MiniLMEncoder:
    """all-MiniLM-L6-v2 as sentence-transformers runs it: transformer, mean pooling over real tokens, unit length"""

    def __init__(self, name: str = MODEL_NAME, revision: str = MODEL_REVISION):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._torch = torch
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
            self.model = AutoModel.from_pretrained(name, revision=revision)
        except OSError as error:
            import os
            raise EmbeddingModelMissing(
                f"the embedding model {name} (revision {revision[:12]}) isn't in the Hugging Face cache "
                f"(HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface')}) and HF_HUB_OFFLINE="
                f"{os.environ.get('HF_HUB_OFFLINE', '0')}. Run `python -m scripts.fetch_embedding_model` "
                f"(the deploy build step does this) with the same HF_HOME. ({type(error).__name__})") from None
        self.model.eval()
        self.max_seq_length = MAX_SEQ_LENGTH

    def encode(self, pieces: list, batch_size: int = BATCH_SIZE) -> np.ndarray:
        """Unit-length float32 vectors, one per piece"""
        torch = self._torch
        out = []
        for start in range(0, len(pieces), batch_size):
            batch = pieces[start:start + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation="longest_first", max_length=self.max_seq_length,
                                 return_tensors="pt")
            with torch.inference_mode():
                hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(pooled, p=2, dim=1).numpy())
        return np.vstack(out).astype(np.float32) if out else np.zeros((0, 384), dtype=np.float32)


def set_torch_threads(default: int = 1) -> int:
    """The server runs the model on TRIAGE_TORCH_THREADS threads (default 1): each torch thread keeps its own
    working memory, and one request's embedding takes milliseconds either way. Returns the number used."""
    import os
    import torch
    raw = os.environ.get(TORCH_THREADS_ENV, str(default)).strip() or str(default)
    try:
        threads = int(raw)
    except ValueError:
        raise ValueError(f"{TORCH_THREADS_ENV} must be a whole number, got {raw!r}") from None
    if threads < 1:
        raise ValueError(f"{TORCH_THREADS_ENV} must be 1 or more, got {threads}")
    torch.set_num_threads(threads)
    return threads


@lru_cache(maxsize=1)
def get_model() -> MiniLMEncoder:
    """Load the model once; later calls return the same object"""
    return MiniLMEncoder()


def is_loaded() -> bool:
    """True once get_model() has loaded the model (for the health check; never loads it)"""
    return get_model.cache_info().currsize > 0

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

def embed_texts(texts: list[str], model=None, batch_size: int = BATCH_SIZE) -> tuple[np.ndarray, list[int]]:
    """(vectors of shape (n, dim), number of chunks per text). Each vector has length 1"""
    model = model or get_model()
    pieces, owner = [], []
    for i, text in enumerate(texts):
        for piece in chunk_text(text or "", model):
            pieces.append(piece)
            owner.append(i)
    
    piece_vecs = model.encode(pieces, batch_size=batch_size)
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
    vecs, _ = embed_texts([text], batch_size=LIVE_BATCH_SIZE)
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