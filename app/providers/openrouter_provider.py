from app.providers.common import require_key
from app.providers.openai_provider import send_openai_format

BASE_URL = "https://openrouter.ai/api/v1"

def send(messages, model_id, **settings):
    return send_openai_format(
        messages,
        model_id,
        provider="openrouter",
        base_url=BASE_URL,
        api_key=require_key("OPENROUTER_API_KEY", "openrouter"),
        **settings,
    )