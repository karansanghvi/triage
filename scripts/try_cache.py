"""Try the cache: the same question twice, then a request that shouldn't be cached."""
import logging

from app.cache import DEFAULT_CACHE_PATH, CachedSender, ResponseCache
from app.reliability import create_client

logging.basicConfig(level=logging.INFO, format="    %(levelname)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)

QUESTION = [{"role": "user", "content": "Name three primary colors. Answer in one line."}]


def show(label, result):
    response = result.response
    print(f"[{label}] {result.model.name}: {response.text.strip()!r}")
    print(
        f"    cached: {response.cached} | latency: {response.latency_ms:.1f} ms | "
        f"reported cost: ${response.reported_cost:.6f} | saved: ${response.cache_saved_cost:.6f}\n"
    )


if __name__ == "__main__":
    # A separate file for this demo, so it doesn't mix with your main cache
    cache = ResponseCache(path=DEFAULT_CACHE_PATH.with_name("demo.sqlite3"))
    sender = CachedSender(cache)
    client = create_client(send_fn=sender)

    print(f"=== Entries already in the demo cache: {cache.count()} ===\n")

    print("=== 1. The same question twice (temperature=0) ===\n")
    show("first ask ", client.send(QUESTION, "medium"))
    show("second ask", client.send(QUESTION, "medium"))

    print("=== 2. temperature=0.7: never cached, both go to the model ===\n")
    show("first ask ", client.send(QUESTION, "medium", temperature=0.7))
    show("second ask", client.send(QUESTION, "medium", temperature=0.7))

    print("=== Cache stats ===\n")
    print(f"    hits: {sender.hits} | misses: {sender.misses} | hit rate: {sender.hit_rate:.0%}")
    print(f"    money saved: ${sender.saved_cost:.6f}")
    print(f"    entries now in the demo cache: {cache.count()}")
    cache.close()