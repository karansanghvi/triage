"""The embedding pipeline the server ran before onnxruntime (torch + transformers), kept as the parity reference.
Test-only: torch and transformers are in requirements-dev.txt, not on the server. Same interface as
app.router.embeddings.MiniLMEncoder (token_ids, decode, encode, max_seq_length), so chunk_text and embed_texts run
unchanged on either."""
import numpy as np

from app.router.embeddings import BATCH_SIZE, DIM, MAX_SEQ_LENGTH, MODEL_NAME, MODEL_REVISION


class TorchMiniLMEncoder:
    def __init__(self, name: str = MODEL_NAME, revision: str = MODEL_REVISION):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModel.from_pretrained(name, revision=revision)
        self.model.eval()
        self.max_seq_length = MAX_SEQ_LENGTH

    def token_ids(self, text: str) -> list:
        return self.tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]

    def decode(self, ids: list) -> str:
        return self.tokenizer.decode(ids)

    def encode(self, pieces: list, batch_size: int = BATCH_SIZE) -> np.ndarray:
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
        return np.vstack(out).astype(np.float32) if out else np.zeros((0, DIM), dtype=np.float32)
