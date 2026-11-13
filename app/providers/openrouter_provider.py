from app.providers.common import server_or_given_key
from app.providers.openai_provider import send_openai_format

BASE_URL = "https://openrouter.ai/api/v1"

def send(messages, model_id, *, api_key=None, **settings):
    """api_key: the caller's own key; None = OPENROUTER_API_KEY from the environment."""
    return send_openai_format(
        messages,
        model_id,
        provider="openrouter",
        base_url=BASE_URL,
        api_key=server_or_given_key(api_key, "OPENROUTER_API_KEY", "openrouter"),
        **settings,
    )