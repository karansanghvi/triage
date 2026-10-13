"""Send one real question to every model, then stream one answer."""
from app.providers.base import send_request
from app.providers.common import ProviderError
from app.registry import load_registry

QUESTION = [
    {"role": "system", "content": "Answer in one short sentence."},
    {"role": "user", "content": "What is the capital of India?"},
]


def try_every_model(registry):
    print("=== Normal requests ===\n")
    for model in registry.values():
        try:
            response = send_request(QUESTION, model)
        except ProviderError as error:
            print(f"[{model.name}] FAILED: {error}\n")
            continue

        estimated = " (estimated)" if response.tokens_estimated else ""
        print(f"[{model.name}] {response.text.strip()!r}")
        print(f"    tokens: {response.input_tokens} in / {response.output_tokens} out{estimated}")
        print(
            f"    latency: {response.latency_ms:.0f} ms | cost: ${response.cost:.6f} | "
            f"reported: ${response.reported_cost:.6f} | finish: {response.finish_reason}\n"
        )


def try_streaming(model):
    print(f"=== Streaming from {model.name} ===\n")
    stream = send_request(
        [{"role": "user", "content": "Count from 1 to 10, with one word after each number."}],
        model,
        stream=True,
    )
    try:
        for piece in stream:
            print(piece, end="", flush=True)
    except ProviderError as error:
        print(f"\nFAILED: {error}")
        return

    response = stream.response
    print(
        f"\n\n    first piece after {response.time_to_first_token_ms or 0:.0f} ms, "
        f"full answer in {response.latency_ms:.0f} ms, "
        f"{response.output_tokens} output tokens"
    )


if __name__ == "__main__":
    registry = load_registry()
    try_every_model(registry)
    try_streaming(registry["groq-small"])