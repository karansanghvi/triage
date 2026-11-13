"""The routing config the API is using right now, updatable without a restart (PUT /v1/routing-config).

Updates are validated with the SAME loaders as startup (quality_settings_from_dict, routing_from_dict) and
swapped in all at once; an invalid update raises ConfigUpdateError and the old config stays. Each request
takes one snapshot at its start, so a request in flight never sees half an update.

Changes live in memory only: config/routing.yaml is never written, and a restart goes back to it.
"""
import copy
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from app.reliability import DEFAULT_ROUTING_PATH, RoutingConfig, routing_from_dict
from app.api.verification import VerificationConfigError, VerificationPolicy, verification_from_dict
from app.router.quality import QualityError, QualitySettings, quality_settings_from_dict

UPDATABLE = ("quality", "fallback_chains")
NOT_PERSISTED_NOTE = ("Changes are kept in memory only: they are not written to config/routing.yaml and are "
                      "lost when the server restarts.")


class ConfigUpdateError(ValueError):
    pass


@dataclass(frozen=True)
class ConfigSnapshot:
    version: int
    raw: dict                       # the parsed routing.yaml with updates applied
    quality: QualitySettings
    routing: RoutingConfig
    verification: VerificationPolicy       # routing.yaml `verification:` (not changeable while running)

    def public(self) -> dict:
        return {"quality": copy.deepcopy(self.raw["quality"]),
                "fallback_chains": copy.deepcopy(self.raw["fallback_chains"])}


class ActiveConfig:
    def __init__(self, raw: dict, registry: dict):
        self.registry = registry
        self._lock = threading.Lock()
        self._snapshot = ConfigSnapshot(1, copy.deepcopy(raw), quality_settings_from_dict(raw),
                                        routing_from_dict(raw, registry), verification_from_dict(raw))

    @classmethod
    def from_file(cls, registry: dict, path: Path = DEFAULT_ROUTING_PATH,
                  server_providers: Optional[list] = None) -> "ActiveConfig":
        """server_providers: replaces routing.yaml's (TRIAGE_SERVER_PROVIDERS, e.g. a host without Ollama)."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if server_providers is not None:
            raw["server_providers"] = list(server_providers)
        return cls(raw, registry)

    def snapshot(self) -> ConfigSnapshot:
        return self._snapshot

    def update(self, patch: dict) -> ConfigSnapshot:
        """Apply a partial update of quality (default, presets) and/or fallback_chains (per tier)."""
        if not isinstance(patch, dict) or not patch:
            raise ConfigUpdateError(f"send a JSON object with one or more of: {', '.join(UPDATABLE)}")
        unknown = set(patch) - set(UPDATABLE)
        if unknown:
            raise ConfigUpdateError(f"can't update {', '.join(sorted(unknown))}: only {', '.join(UPDATABLE)} "
                                    "can be changed while running")
        with self._lock:
            current = self._snapshot
            raw = copy.deepcopy(current.raw)
            if "quality" in patch:
                q = patch["quality"]
                if not isinstance(q, dict) or not q or set(q) - {"default", "presets"}:
                    raise ConfigUpdateError("quality must be an object with 'default' and/or 'presets'")
                raw["quality"] = {**raw.get("quality", {}), **copy.deepcopy(q)}
            if "fallback_chains" in patch:
                chains = patch["fallback_chains"]
                if not isinstance(chains, dict) or not chains:
                    raise ConfigUpdateError("fallback_chains must be an object of tier -> list of model names")
                for tier, names in chains.items():
                    old = current.raw["fallback_chains"].get(tier)
                    if old is not None and isinstance(names, list) and names and names[0] != old[0]:
                        raise ConfigUpdateError(
                            f"fallback_chains.{tier}: the first model must stay '{old[0]}'. The classifier predicts "
                            f"whether that exact model will pass, so routing to a different first model would make "
                            f"its predictions meaningless. Only the backup models after it may change.")
                raw["fallback_chains"] = {**raw["fallback_chains"], **copy.deepcopy(chains)}
            try:
                quality = quality_settings_from_dict(raw, "routing config update")
                routing = routing_from_dict(raw, self.registry)
            except (QualityError, ValueError) as error:
                raise ConfigUpdateError(str(error)) from None
            self._snapshot = ConfigSnapshot(current.version + 1, raw, quality, routing, current.verification)
            return self._snapshot
