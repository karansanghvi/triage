"""Phase 1 final check: real requests everywhere, then a real outage (Ollama turned off).

Run with:  python -m scripts.final_check
"""
import logging
import os
import time
import urllib.request

from app.providers.base import send_request
from app.providers.common import ProviderError
from app.registry import load_registry
from app.reliability import AllModelsFailed, create_client

# Show only Triage's own messages (retries, fallbacks, circuit breaker),
# and hide the HTTP libraries' chatter.
logging.basicConfig(level=logging.WARNING, format="    %(levelname)s: %(message)s")
logging.getLogger("triage").setLevel(logging.INFO)

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").removesuffix("/v1")
QUESTION = [{"role": "user", "content": "Reply with just the word: ready"}]

results = []  # (check name, passed)


def record(name, passed, detail=""):
    results.append((name, passed))
    extra = f" ({detail})" if detail else ""
    print(f"  {'PASS' if passed else 'FAIL'}: {name}{extra}")


def ollama_is_running() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2):
            return True
    except OSError:
        return False


def wait_until(prompt, condition, problem):
    """Ask the user to do something, then check it really happened."""
    while True:
        input(f"\n>>> {prompt} Then press Enter... ")
        if condition():
            return
        print(f"    Hmm, {problem}. Try again.")


def show_path(result):
    return " -> ".join(
        f"{a.model_name} ({a.outcome}{': ' + a.error_kind if a.error_kind else ''})"
        for a in result.attempts
    )


# ---------- Part 1 ----------

def check_every_model(registry):
    print("=== 1. Every model answers through send_request() ===\n")
    for model in registry.values():
        try:
            response = send_request(QUESTION, model)
        except ProviderError as error:
            record(f"{model.name} answers", False, str(error)[:120])
            continue
        text = response.text.strip()
        record(f"{model.name} answers", bool(text), f"{response.latency_ms:.0f} ms, {text[:30]!r}")


# ---------- Part 2 ----------

def check_every_tier(client):
    print("\n=== 2. Every tier answers through the reliability layer ===\n")
    for tier in ("low", "medium", "high"):
        try:
            result = client.send(QUESTION, tier)
        except AllModelsFailed as error:
            record(f"{tier} tier answers", False, str(error)[:120])
            continue
        note = " after a fallback" if result.fell_back else ""
        record(f"{tier} tier answers", True, f"by {result.model.name}{note}")


# ---------- Part 3 ----------

def check_outage(client):
    print("\n=== 3. A real outage: Ollama turned off ===")
    print("    Quit Ollama now: click the llama icon in the menu bar and choose Quit Ollama")
    print("    (or press Ctrl+C in the terminal window running `ollama serve`).")
    wait_until("Quit Ollama.", lambda: not ollama_is_running(), "Ollama still seems to be running")
    print("\n    Ollama is off. Sending 4 low-tier requests...\n")

    answered = []
    for number in range(1, 5):
        question = [{"role": "user", "content": f"What is {number} plus {number}? Answer with just the number."}]
        print(f"  [request {number}]")
        try:
            result = client.send(question, "low")
        except AllModelsFailed as error:
            print(f"    FAILED: {error}\n")
            continue
        print(f"    answered by {result.model.name}: {result.response.text.strip()!r}")
        print(f"    path: {show_path(result)}\n")
        answered.append(result)

    record(
        "every request survives by falling back to another provider",
        len(answered) == 4 and all(r.model.provider != "ollama" for r in answered),
    )
    record(
        "the circuit breaker skips Ollama once it's clearly down",
        bool(answered) and answered[-1].attempts[0].outcome == "skipped",
    )


# ---------- Part 4 ----------

def check_recovery(client):
    print("\n=== 4. Recovery: Ollama comes back ===")
    wait_until(
        "Open Ollama again (or run `ollama serve`).",
        ollama_is_running,
        "Ollama doesn't answer yet; give it a few seconds",
    )

    seconds_left = client.breaker.seconds_left("ollama")
    if seconds_left > 0:
        print(f"\n    The circuit breaker still skips Ollama for {seconds_left:.0f}s. Waiting it out...")
        time.sleep(seconds_left + 1)

    print("    Sending one more low-tier request...\n")
    try:
        result = client.send([{"role": "user", "content": "Reply with just the word: back"}], "low")
    except AllModelsFailed as error:
        record("Ollama is used again after the cooldown", False, str(error)[:120])
        return
    print(f"    path: {show_path(result)}")
    record("Ollama is used again after the cooldown", result.model.provider == "ollama",
           f"answered by {result.model.name}")


# ---------- Summary ----------

def print_summary():
    passed = sum(1 for _, ok in results if ok)
    print(f"\n=== Phase 1 summary: {passed} of {len(results)} checks passed ===\n")
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'} {name}")
    if passed == len(results):
        print("\n  Phase 1 is done: one function calls any model, and a provider")
        print("  outage falls back to another provider automatically.")
    else:
        print("\n  Some checks failed. If a cloud provider was busy (server or timeout")
        print("  errors), wait a few minutes and run this again.")


if __name__ == "__main__":
    registry = load_registry()
    # No cache on purpose: a saved answer could hide the outage we're testing
    client = create_client()
    try:
        check_every_model(registry)
        check_every_tier(client)
        check_outage(client)
        check_recovery(client)
    except KeyboardInterrupt:
        print("\n\n    Stopped early.")
    print_summary()