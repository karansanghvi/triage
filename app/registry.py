from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

# Project root = two folders above this file (triage/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "models.yaml"

VALID_PROVIDERS = {"groq", "gemini", "ollama", "openrouter", "openai", "anthropic"}
VALID_TIERS = {"low", "medium", "high"}
TOKENS_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class PriceAs:
    """The realistic paid price a free model stands in for."""
    label: str
    input_price: float
    output_price: float


@dataclass(frozen=True)
class ModelConfig:
    """Everything Triage needs to know about one model."""
    name: str
    provider: str
    model_id: str
    tier: str
    input_price: float          # dollars per 1M input tokens (what you pay)
    output_price: float         # dollars per 1M output tokens (what you pay)
    avg_latency_ms: float = 0.0
    price_as: Optional[PriceAs] = None
    max_output_tokens: Optional[int] = None   # hard cap for this model, if it has a low limit

    def actual_cost(self, input_tokens: int, output_tokens: int) -> float:
        """What this request really cost you."""
        return (
            input_tokens * self.input_price + output_tokens * self.output_price
        ) / TOKENS_PER_MILLION

    def reported_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Realistic cost for savings math: uses price_as when there is one."""
        if self.price_as is None:
            return self.actual_cost(input_tokens, output_tokens)
        return (
            input_tokens * self.price_as.input_price
            + output_tokens * self.price_as.output_price
        ) / TOKENS_PER_MILLION


# ---------- Validation helpers ----------

def _require(entry: dict, field: str, where: str):
    """Get a required field, or raise a clear error."""
    if field not in entry or entry[field] is None or entry[field] == "":
        raise ValueError(f"{where}: missing required field '{field}'")
    value = entry[field]
    if isinstance(value, str) and "REPLACE_ME" in value:
        raise ValueError(f"{where}: field '{field}' still says REPLACE_ME")
    return value


def _parse_price(value, field: str, where: str) -> float:
    """Prices must be numbers that are zero or more."""
    if isinstance(value, str) and "REPLACE_ME" in value:
        raise ValueError(f"{where}: field '{field}' still says REPLACE_ME")
    # bool is a kind of int in Python, so rule it out explicitly
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: '{field}' must be a number, got {value!r}")
    if value < 0:
        raise ValueError(f"{where}: '{field}' can't be negative")
    return float(value)


def _parse_optional_whole_number(value, field: str, where: str) -> Optional[int]:
    """Optional fields like max_output_tokens: a whole number above 0, or left out."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where}: '{field}' must be a whole number above 0, got {value!r}")
    return value


def _parse_model(entry, index: int) -> ModelConfig:
    """Turn one YAML entry into a checked ModelConfig."""
    if not isinstance(entry, dict):
        raise ValueError(f"model #{index + 1}: each model must be a set of fields")

    name = str(_require(entry, "name", f"model #{index + 1}"))
    where = f"model '{name}'"

    provider = str(_require(entry, "provider", where))
    if provider not in VALID_PROVIDERS:
        raise ValueError(
            f"{where}: unknown provider '{provider}'. "
            f"Use one of: {', '.join(sorted(VALID_PROVIDERS))}"
        )

    tier = str(_require(entry, "tier", where))
    if tier not in VALID_TIERS:
        raise ValueError(
            f"{where}: unknown tier '{tier}'. Use one of: low, medium, high"
        )

    price_as = None
    raw_price_as = entry.get("price_as")
    if raw_price_as is not None:
        if not isinstance(raw_price_as, dict):
            raise ValueError(f"{where}: 'price_as' must be a set of fields")
        price_where = f"{where} price_as"
        price_as = PriceAs(
            label=str(_require(raw_price_as, "label", price_where)),
            input_price=_parse_price(
                _require(raw_price_as, "input_price", price_where),
                "input_price", price_where,
            ),
            output_price=_parse_price(
                _require(raw_price_as, "output_price", price_where),
                "output_price", price_where,
            ),
        )

    return ModelConfig(
        name=name,
        provider=provider,
        model_id=str(_require(entry, "model_id", where)),
        tier=tier,
        input_price=_parse_price(_require(entry, "input_price", where), "input_price", where),
        output_price=_parse_price(_require(entry, "output_price", where), "output_price", where),
        avg_latency_ms=_parse_price(entry.get("avg_latency_ms", 0), "avg_latency_ms", where),
        price_as=price_as,
        max_output_tokens=_parse_optional_whole_number(
            entry.get("max_output_tokens"), "max_output_tokens", where
        ),
    )


# ---------- Public functions ----------

def load_registry(path=DEFAULT_CONFIG_PATH) -> dict[str, ModelConfig]:
    """Read models.yaml and return {name: ModelConfig}."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model registry not found at {path}")

    with path.open() as file:
        data = yaml.safe_load(file) or {}

    raw_models = data.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError(f"{path.name} must contain a non-empty 'models' list")

    registry: dict[str, ModelConfig] = {}
    for index, entry in enumerate(raw_models):
        model = _parse_model(entry, index)
        if model.name in registry:
            raise ValueError(f"Duplicate model name '{model.name}' in {path.name}")
        registry[model.name] = model

    return registry


def models_in_tier(registry: dict[str, ModelConfig], tier: str) -> list[ModelConfig]:
    """All models in one tier, e.g. every 'low' model."""
    return [model for model in registry.values() if model.tier == tier]


if __name__ == "__main__":
    registry = load_registry()
    print(f"Loaded {len(registry)} models\n")
    print("Example cost for 1,000 input + 500 output tokens:\n")
    for model in registry.values():
        cost = model.reported_cost(1_000, 500)
        print(f"  {model.name:<14} {model.provider:<8} tier={model.tier:<7} ${cost:.6f}")