import pytest 

from app.registry import load_registry, models_in_tier 

VALID_YAML = """
models:
  - name: cheap
    provider: groq
    model_id: small-model
    tier: low
    input_price: 0
    output_price: 0
    price_as:
      label: Some paid model
      input_price: 1.0
      output_price: 2.0
  - name: strong
    provider: gemini
    model_id: big-model
    tier: high
    input_price: 3.0
    output_price: 6.0
"""

def write_config(tmp_path, text):
    path = tmp_path / "models.yaml"
    path.write_text(text)
    return path