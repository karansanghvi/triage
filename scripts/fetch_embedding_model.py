"""Download the router's embedding model into the Hugging Face cache, for a deploy build step.

    HF_HOME=/path/inside/the/app python -m scripts.fetch_embedding_model

The server starts with HF_HUB_OFFLINE=1 and loads the model from that cache, so it never needs the network for it.
Downloads exactly app.router.embeddings.MODEL_FILES (the fp32 ONNX export and its tokenizer.json), all at
MODEL_REVISION (the revision the classifier was evaluated with), then loads them offline once to prove the cache
is complete.
"""
import os
import subprocess
import sys

from app.router.embeddings import MODEL_FILES, MODEL_NAME, MODEL_REVISION


def main() -> None:
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL_NAME, revision=MODEL_REVISION, allow_patterns=list(MODEL_FILES))
    print(f"downloaded {MODEL_NAME}@{MODEL_REVISION[:12]} ({', '.join(MODEL_FILES)}) to {path}")
    # Prove it loads with the network off, in a fresh process (as the server will)
    check = ("from app.router.embeddings import get_model; m = get_model(); "
             "v = m.encode(['hello']); assert v.shape == (1, 384), v.shape; print('offline load ok')")
    subprocess.run([sys.executable, "-c", check], env={**os.environ, "HF_HUB_OFFLINE": "1"}, check=True)


if __name__ == "__main__":
    main()
