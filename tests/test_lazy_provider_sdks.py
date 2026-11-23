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


def test_importing_the_app_loads_no_provider_sdk():
    out = _run("import sys; import app.api.main; "
               "print([m for m in ('openai', 'anthropic', 'google.genai', 'httpx2') if m in sys.modules])")
    assert out == "[]"


def test_after_startup_the_server_sdks_load_in_the_background_and_anthropic_waits_for_use():
    out = _run("""
import sys, time
from fastapi.testclient import TestClient
from app.api import main
from tests.api_helpers import make_deps
with TestClient(main.create_app(deps=make_deps(), public=False, accounts=False)) as client:
    assert client.get('/healthz').status_code in (200, 503)
    for _ in range(300):
        if 'google.genai' in sys.modules and 'openai' in sys.modules: break
        time.sleep(0.1)
    print(['openai' in sys.modules, 'google.genai' in sys.modules, 'anthropic' in sys.modules])
""")
    assert out == "[True, True, False]"


def test_a_lazy_module_imports_on_first_use_and_passes_attributes_through(monkeypatch):
    lazy = LazyModule("json")
    assert "not imported yet" in repr(lazy)
    assert lazy.dumps({"a": 1}) == '{"a": 1}'
    monkeypatch.setattr(lazy, "dumps", lambda obj: "patched")         # tests patch SDK attributes like this
    import json
    assert json.dumps({}) == "patched"


def test_a_request_that_needs_an_sdk_first_waits_for_the_background_import():
    """Two threads importing the same module: the second waits for the first (Python's import lock), and both see
    the finished module."""
    code = """
import sys, threading, importlib
from app.providers.lazy import warm_in_background, LazyModule
thread = warm_in_background(("openai",))
openai = LazyModule("openai")
cls = openai.OpenAI                     # a request needing it now: waits rather than failing
thread.join()
print(cls is sys.modules["openai"].OpenAI)
"""
    assert _run(code) == "True"


def test_a_failed_background_import_is_logged_not_raised(caplog):
    with caplog.at_level("WARNING"):
        warm_in_background(("no_such_module_for_triage",)).join()
    assert "background import of no_such_module_for_triage failed" in caplog.text
