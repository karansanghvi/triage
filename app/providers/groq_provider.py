from app.providers.common import require_key
from app.providers.openai_provider import send_openai_format

BASE_URL = "https://api.groq.com/openai/v1"

def send(messages, model_id, **settings):
    return send_openai_format(
        messages,
        model_id,
        provider="groq",
        base_url=BASE_URL,
        api_key=require_key("GROQ_API_KEY", "groq"),
        **settings,
    )