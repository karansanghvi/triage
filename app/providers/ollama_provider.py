import os 
from app.providers.openai_provider import send_openai_format

DEFAULT_BASE_URL = "http://localhost:11434/v1"

def send(messages, model_id, **settings):
    return send_openai_format(
        messages,
        model_id,
        provider="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
        api_key="ollama", # Ollama ignores the key, but the library needs a value 
        **settings,
    )