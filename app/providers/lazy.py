"""Provider SDKs, imported when first used instead of when the server starts.

The SDKs are the slowest part of starting the API (measured at a tenth of a CPU: anthropic about 20 s, openai and
google-genai about 9 s each, under -X importtime). A LazyModule stands in for a module and imports it on first
attribute access, so `anthropic.Anthropic(...)` and `isinstance(error, openai.APIError)` work unchanged. Setting an
attribute (tests' monkeypatch) sets it on the real module.

    anthropic   imported on the first Anthropic request (it's bring-your-own-key only, never a server provider)
    openai,     imported in a background thread right after the server starts listening (warm_in_background,
    google-genai called by the app's startup): /healthz answers sooner, and a request that needs one before it's
                ready waits for it (Python's import lock), rather than failing
"""
import importlib
import logging
import threading
import time

logger = logging.getLogger("triage.providers")

# What the server's own providers need (TRIAGE_SERVER_PROVIDERS=groq,gemini: Groq speaks OpenAI's format)
SERVER_SDKS = ("openai", "google.genai", "google.genai.types", "google.genai.errors")


class LazyModule:
    """Imports `name` on first attribute access, then behaves like the module"""

    def __init__(self, name: str):
        object.__setattr__(self, "_lazy_name", name)
        object.__setattr__(self, "_lazy_module", None)

    def _lazy_load(self):
        module = object.__getattribute__(self, "_lazy_module")
        if module is None:
            module = importlib.import_module(object.__getattribute__(self, "_lazy_name"))
            object.__setattr__(self, "_lazy_module", module)
        return module

    def __getattr__(self, attr):
        return getattr(self._lazy_load(), attr)

    def __setattr__(self, attr, value):
        setattr(self._lazy_load(), attr, value)

    def __delattr__(self, attr):
        delattr(self._lazy_load(), attr)

    def __repr__(self):
        loaded = object.__getattribute__(self, "_lazy_module") is not None
        return f"<lazy module {object.__getattribute__(self, '_lazy_name')!r}{'' if loaded else ' (not imported yet)'}>"


def _import_all(names) -> None:
    start = time.monotonic()
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as error:          # the request that needs it will raise the real error
            logger.warning("background import of %s failed (%s)", name, type(error).__name__)
    logger.info("startup: provider SDKs (%s) imported in the background in %.1fs", ", ".join(names),
                time.monotonic() - start)


def warm_in_background(names=SERVER_SDKS) -> threading.Thread:
    """Import `names` in a daemon thread and return it (the server keeps serving meanwhile)"""
    thread = threading.Thread(target=_import_all, args=(tuple(names),), name="import-provider-sdks", daemon=True)
    thread.start()
    return thread
