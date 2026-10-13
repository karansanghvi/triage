"""Try the reliability layer: real requests per tier, then a simulated outage."""
import logging

from app.providers.base import send_request
from app.providers.common import ProviderError
from app.reliability import AllModelsFailed, create_client

# Show the reliability layer's log messages (retries, fallbacks, circuit breaker)
logging.basicConfig(level=logging.INFO, format="    %(levelname)s: %(message)s")

QUESTION = [{"role": "user", "content": "What is 12 times 12? Answer with just the number."}]


def show(result):
    path = " -> ".join(
        f"{a.model_name} ({a.outcome}{': ' + a.error_kind if a.error_kind else ''})"
        for a in result.attempts
    )
    print(f"  answered by {result.model.name}: {result.response.text.strip()!r}")
    print(f"  path: {path}")
    print(f"  fell back: {result.fell_back}\n")


def real_requests():
    print("=== 1. Real requests, one per tier ===\n")
    client = create_client()
    for tier in ("low", "medium", "high"):
        print(f"[{tier} tier]")
        try:
            show(client.send(QUESTION, tier))
        except AllModelsFailed as error:
            print(f"  FAILED: {error}\n")


def simulated_outage(down_provider="groq"):
    print(f"=== 2. Simulated outage: every {down_provider} call fails ===\n")

    def flaky_send(messages, model, **settings):
        # Pretend this provider is down; everything else is real
        if model.provider == down_provider:
            raise ProviderError(down_provider, "server", "simulated outage")
        return send_request(messages, model, **settings)

    client = create_client(send_fn=flaky_send)
    for number in (1, 2):
        print(f"[medium tier, request {number}]")
        try:
            show(client.send(QUESTION, "medium"))
        except AllModelsFailed as error:
            print(f"  FAILED: {error}\n")

    print("Fallback log:")
    for event in client.fallback_events:
        print(f"  {event.tier} tier: {event.from_model} -> {event.to_model} ({event.reason})")


if __name__ == "__main__":
    real_requests()
    simulated_outage()