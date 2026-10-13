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


def test_loads_valid_config(tmp_path):
    registry = load_registry(write_config(tmp_path, VALID_YAML))
    assert set(registry) == {"cheap", "strong"}
    assert registry["cheap"].tier == "low"


def test_free_model_uses_price_as_for_reported_cost(tmp_path):
    cheap = load_registry(write_config(tmp_path, VALID_YAML))["cheap"]
    # 1M input tokens at $1 + 1M output tokens at $2 = $3
    assert cheap.actual_cost(1_000_000, 1_000_000) == 0
    assert cheap.reported_cost(1_000_000, 1_000_000) == pytest.approx(3.0)


def test_paid_model_without_price_as_uses_real_price(tmp_path):
    strong = load_registry(write_config(tmp_path, VALID_YAML))["strong"]
    assert strong.reported_cost(1_000_000, 1_000_000) == pytest.approx(9.0)


def test_models_in_tier(tmp_path):
    registry = load_registry(write_config(tmp_path, VALID_YAML))
    assert [m.name for m in models_in_tier(registry, "low")] == ["cheap"]


def test_rejects_unknown_tier(tmp_path):
    bad = VALID_YAML.replace("tier: low", "tier: lwo")
    with pytest.raises(ValueError, match="unknown tier"):
        load_registry(write_config(tmp_path, bad))


def test_rejects_duplicate_names(tmp_path):
    bad = VALID_YAML.replace("name: strong", "name: cheap")
    with pytest.raises(ValueError, match="Duplicate"):
        load_registry(write_config(tmp_path, bad))


def test_rejects_leftover_placeholder(tmp_path):
    bad = VALID_YAML.replace("model_id: small-model", "model_id: REPLACE_ME")
    with pytest.raises(ValueError, match="REPLACE_ME"):
        load_registry(write_config(tmp_path, bad))


def test_real_config_loads():
    """Your actual config/models.yaml should load without errors."""
    registry = load_registry()
    assert len(registry) > 0