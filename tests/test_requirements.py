"""The server's packages (requirements.txt -> requirements.lock) against the development extras
(requirements-dev.txt): every server pin is locked at the same version, and nothing that only training, evaluation
or tests use reaches the server."""
import re

from app.registry import PROJECT_ROOT

DEV_ONLY = {"torch", "transformers", "sentence-transformers", "datasets", "pytest", "pandas", "pyarrow"}


def _pins(name):
    pins = {}
    for line in (PROJECT_ROOT / name).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?==(\S+)", line)
        if match:
            pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    return pins


def test_every_server_pin_is_locked_at_the_same_version():
    lock = _pins("requirements.lock")
    for name, version in _pins("requirements.txt").items():
        assert lock.get(name) == version, name


def test_nothing_dev_only_reaches_the_server():
    for name in ("requirements.txt", "requirements.lock"):
        assert not DEV_ONLY & set(_pins(name)), name
    assert "extra-index-url" not in (PROJECT_ROOT / "requirements.lock").read_text()   # no PyTorch index any more


def test_the_server_runs_the_embedding_model_on_onnxruntime():
    lock = _pins("requirements.lock")
    assert lock["onnxruntime"] == "1.31.1" and "tokenizers" in lock and "huggingface-hub" in lock


def test_the_dev_file_adds_to_the_server_file():
    text = (PROJECT_ROOT / "requirements-dev.txt").read_text()
    assert re.search(r"^-r requirements\.txt$", text, re.M)
    dev = _pins("requirements-dev.txt")
    assert {"torch", "transformers", "sentence-transformers", "datasets", "pytest"} <= set(dev)
    assert not set(dev) & set(_pins("requirements.txt"))               # each package pinned in one place
