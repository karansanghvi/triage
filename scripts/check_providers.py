import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Each provider: where to send requests, and which .env key to use.
# All three accept OpenAI's format, so one library covers them all.
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
    },
    "ollama": {
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "key_env": None,  # Ollama runs locally and needs no key
    },
}

MAX_TO_SHOW = 15


def check_provider(name, config):
    # Get the key, or skip this provider if it's missing
    if config["key_env"]:
        api_key = os.getenv(config["key_env"])
        if not api_key:
            print(f"[{name}] SKIPPED: {config['key_env']} is missing in .env\n")
            return
    else:
        # Ollama ignores the key, but the library requires some value
        api_key = "ollama"

    client = OpenAI(base_url=config["base_url"], api_key=api_key, timeout=15)

    # Listing models tests the key without using any tokens
    try:
        response = client.models.list()
        model_ids = sorted(model.id for model in response.data)
    except Exception as error:
        print(f"[{name}] FAILED: {type(error).__name__}: {error}\n")
        return

    print(f"[{name}] OK: {len(model_ids)} models available")
    for model_id in model_ids[:MAX_TO_SHOW]:
        print(f"    {model_id}")
    if len(model_ids) > MAX_TO_SHOW:
        print(f"    ...and {len(model_ids) - MAX_TO_SHOW} more")
    print()


if __name__ == "__main__":
    print("Checking providers...\n")
    for name, config in PROVIDERS.items():
        check_provider(name, config)