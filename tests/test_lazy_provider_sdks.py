"""Provider SDKs load after startup, not during it (app/providers/lazy.py): the API answers /healthz sooner."""
import subprocess
import sys
import threading

import pytest

from app.providers.lazy import LazyModule, warm_in_background
from app.registry import PROJECT_ROOT


def _run(code: str) -> str:
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True,
                            env={**__import__("os").environ, "HF_HUB_OFFLINE": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip().splitlines()[-1]


def test_importing_the_app_doesnt_load_the_anthropic_sdk():
    out = _run("import sys; import app.api.main; print([m for m in ('anthropic',) if m in sys.modules])")
    assert out == "[]"


def test_a_lazy_module_imports_on_first_use_and_passes_attributes_through(monkeypatch):
    lazy = LazyModule("json")
    assert "not imported yet" in repr(lazy)
    assert lazy.dumps({"a": 1}) == '{"a": 1}'
    monkeypatch.setattr(lazy, "dumps", lambda obj: "patched")         # tests patch SDK attributes like this
    import json
    assert json.dumps({}) == "patched"
