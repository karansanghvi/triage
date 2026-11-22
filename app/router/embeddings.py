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

The model runs on onnxruntime (the model repo's own fp32 export, onnx/model.onnx, at the pinned revision), with
the `tokenizers` library and numpy: tokenizer -> transformer -> mean pooling -> L2 normalisation, exactly the
all-MiniLM-L6-v2 sentence-transformers pipeline (modules.json: Transformer, Pooling(mean), Normalize). No torch or
transformers at runtime (about 300 MB of memory less); they are test-only now. tests/test_embedding_parity.py
checks this against the torch/transformers pipeline and against sentence-transformers itself.

The tokenizer.json in the model repo has truncation and padding at 128 tokens built in. The encoder sets both
itself (truncate at MAX_SEQ_LENGTH = 256, pad to the longest in each batch), as sentence-transformers does.
 
Usage (batch, for training):
    python -m app.router.embeddings --input data/prompts.jsonl --output data/embeddings.npz
 
Usage (in code, at request time):
    from app.router.embeddings import embed_one
    vec = embed_one("Summarize this email: ...")      # numpy array, shape (384,)
"""

from __future__ import annotations

import argparse 
import json 
import os
import threading
import time 
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.router.features import prompt_text

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# The exact revision the classifier was evaluated with; the build downloads this one (scripts/fetch_embedding_model)
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
# Files the server needs, all at MODEL_REVISION: the fp32 ONNX export and the tokenizer that goes with it
ONNX_FILE = "onnx/model.onnx"
TOKENIZER_FILE = "tokenizer.json"
MODEL_FILES = (ONNX_FILE, TOKENIZER_FILE)
MAX_SEQ_LENGTH = 256    # the model's sentence_bert_config.json: longer input is cut, so chunk_text splits first
DIM = 384
BATCH_SIZE = 32         # batch (training) runs
LIVE_BATCH_SIZE = 4     # live requests: a long prompt's chunks go through a few at a time, to keep memory low
CHUNK_MARGIN = 16   # tokens kept free in each chunk (see chunk_text)
THREADS_ENV = "TRIAGE_EMBEDDING_THREADS"

# One embedding at a time in this process. A long prompt's embedding holds about 55 MB of working memory while it
# runs; on a 512 MB server (Render's free plan) two at once could push the API past its limit and get it killed.
# Running them one after another costs nothing on a small CPU share, where two at once wouldn't finish sooner.
EMBED_LOCK = threading.Lock()

# onnxruntime's official builds send usage telemetry to Microsoft by default (its Privacy.md). This turns all of it
# off (no uploader, no events, no device identifier) and must be set before onnxruntime first loads, so it is set
# here, before this module ever imports it. Forced, not a default: the privacy policy says Triage shares nothing
# with anyone beyond the model providers. The Dockerfile sets it too.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"


class EmbeddingModelMissing(RuntimeError):
    """The model files aren't in the Hugging Face cache and downloading is off (a deployed server)"""


def model_file(filename: str, name: str = MODEL_NAME, revision: str = MODEL_REVISION) -> str:
    """Local path of one model file at `revision`, from the Hugging Face cache (downloaded first unless
    HF_HUB_OFFLINE=1). Missing: EmbeddingModelMissing, naming the fetch script."""
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(name, filename, revision=revision)
    except Exception as error:
        raise EmbeddingModelMissing(
            f"the embedding model {name} (revision {revision[:12]}, {filename}) isn't in the Hugging Face cache "
            f"(HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface')}) and HF_HUB_OFFLINE="
            f"{os.environ.get('HF_HUB_OFFLINE', '0')}. Run `python -m scripts.fetch_embedding_model` "
            f"(the deploy build step does this) with the same HF_HOME. ({type(error).__name__})") from None


def embedding_threads(default: int = 1) -> int:
    """TRIAGE_EMBEDDING_THREADS (default 1): onnxruntime's threads for the model. One request's embedding takes
    milliseconds either way, and each extra thread keeps its own working memory. Bad values raise ValueError."""
    raw = os.environ.get(THREADS_ENV, str(default)).strip() or str(default)
    try:
        threads = int(raw)
    except ValueError:
        raise ValueError(f"{THREADS_ENV} must be a whole number, got {raw!r}") from None
    if threads < 1:
        raise ValueError(f"{THREADS_ENV} must be 1 or more, got {threads}")
    return threads


class MiniLMEncoder:
    """all-MiniLM-L6-v2 as sentence-transformers runs it: transformer, mean pooling over real tokens, unit length.
    onnxruntime on the CPU, fp32, TRIAGE_EMBEDDING_THREADS threads."""

    def __init__(self, name: str = MODEL_NAME, revision: str = MODEL_REVISION, threads: int | None = None):
        os.environ["ORT_DISABLE_TELEMETRY"] = "1"          # in case something reset it after this module loaded
        import onnxruntime as ort
        from tokenizers import Tokenizer
        ort.disable_telemetry_events()
        onnx_path = model_file(ONNX_FILE, name, revision)
        tokenizer_path = model_file(TOKENIZER_FILE, name, revision)
        self.max_seq_length = MAX_SEQ_LENGTH
        # For chunking: the whole text, no special tokens added, no cut, no padding
        self.raw_tokenizer = Tokenizer.from_file(tokenizer_path)
        self.raw_tokenizer.no_truncation()
        self.raw_tokenizer.no_padding()
        # For the model: cut at MAX_SEQ_LENGTH, padded to the longest in the batch (overriding the file's 128)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(MAX_SEQ_LENGTH, strategy="longest_first")
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads or embedding_threads()
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(onnx_path, options, providers=["CPUExecutionProvider"])
        self.threads = options.intra_op_num_threads

    def token_ids(self, text: str) -> list:
        """The text's token ids, without special tokens and uncut"""
        return self.raw_tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list) -> str:
        return self.raw_tokenizer.decode(ids, skip_special_tokens=False)

    def encode(self, pieces: list, batch_size: int = BATCH_SIZE) -> np.ndarray:
        """Unit-length float32 vectors, one per piece"""
        out = []
        for start in range(0, len(pieces), batch_size):
            encodings = self.tokenizer.encode_batch(pieces[start:start + batch_size])
            ids = np.array([e.ids for e in encodings], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
            types = np.array([e.type_ids for e in encodings], dtype=np.int64)
            hidden = self.session.run(["last_hidden_state"],
                                      {"input_ids": ids, "attention_mask": mask, "token_type_ids": types})[0]
            weights = mask[..., None].astype(hidden.dtype)
            pooled = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
            norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)   # as F.normalize
            out.append(pooled / norms)
        return np.vstack(out).astype(np.float32) if out else np.zeros((0, DIM), dtype=np.float32)


@lru_cache(maxsize=1)
def get_model() -> MiniLMEncoder:
    """Load the model once; later calls return the same object"""
    return MiniLMEncoder()


def is_loaded() -> bool:
    """True once get_model() has loaded the model (for the health check; never loads it)"""
    return get_model.cache_info().currsize > 0

def chunk_text(text: str, model) -> list[str]:
    """Split text into pieces the model can read whole. Short text comes back as a single piece."""
    # Room for the model's own start/end tokens, plus a margin: a decoded chunk can re-tokenize a few
    # tokens longer, and the model would silently drop the overflow
    size = model.max_seq_length - 2 - CHUNK_MARGIN
    ids = model.token_ids(text)
    if len(ids) <= size:
        return [text]
    return [model.decode(ids[i:i + size]) for i in range(0, len(ids), size)]

def embed_texts(texts: list[str], model=None, batch_size: int = BATCH_SIZE) -> tuple[np.ndarray, list[int]]:
    """(vectors of shape (n, dim), number of chunks per text). Each vector has length 1"""
    model = model or get_model()
    pieces, owner = [], []
    for i, text in enumerate(texts):
        for piece in chunk_text(text or "", model):
            pieces.append(piece)
            owner.append(i)
    
    with EMBED_LOCK:                    # one embedding at a time (memory; see EMBED_LOCK)
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