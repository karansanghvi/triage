"""Everything a request needs, built once at startup: predictor, ReliableClient, config, cache, log."""
import os
from dataclasses import dataclass, field
from typing import Optional

from app.api.config_state import ActiveConfig
from app.api.request_log import InMemoryRequestLog, SQLiteRequestLog, hash_secret_from_env, open_request_log
from app.cache import DEFAULT_CACHE_PATH, CachedSender, ResponseCache
from app.providers.base import send_request
from app.registry import load_registry
from app.reliability import ReliableClient
from app.verifier.ai_checker import CheckerSettings, load_checker_settings
from app.verifier.judge import JudgeSettings, load_judge_settings

FAKE_MODELS_ENV, CACHE_ENV, CACHE_PATH_ENV = "TRIAGE_FAKE_MODELS", "TRIAGE_CACHE", "TRIAGE_CACHE_PATH"


@dataclass
class RouterDeps:
    registry: dict
    config: ActiveConfig
    client: ReliableClient
    predictor: object                     # app.router.predictor.Predictor (or a test fake with predict_messages)
    log: object                           # a request_log.RequestLog
    classifier: str = ""                  # "name v<version>", for the log
    cached_sender: Optional[CachedSender] = None
    fake_models: Optional[object] = None
    hash_secret: bytes = field(default=b"", repr=False)   # TRIAGE_HASH_SECRET: keys prompt_hmac; never logged
    checker_settings: Optional[CheckerSettings] = None   # config/checker.yaml (sync verification)
    judge_settings: Optional[JudgeSettings] = None       # config/judge.yaml (background labels)
    labeler: Optional[object] = None                     # background.BackgroundLabeler

    def strong_model(self) -> str:
        return self.config.snapshot().routing.chains["high"][0]

    def update_config(self, patch: dict):
        snapshot = self.config.update(patch)          # raises ConfigUpdateError and changes nothing if invalid
        self.client.routing = snapshot.routing        # the breaker (and its memory of failures) is kept
        return snapshot


def build_deps(predictor, send_fn, *, registry=None, config: Optional[ActiveConfig] = None,
               cache: Optional[ResponseCache] = None, log=None, classifier: str = "", sleep=None,
               fake_models=None, hash_secret: Optional[bytes] = None, labeler=None,
               checker_settings: Optional[CheckerSettings] = None,
               judge_settings: Optional[JudgeSettings] = None) -> RouterDeps:
    registry = registry or load_registry()
    config = config or ActiveConfig.from_file(registry)
    cached = CachedSender(cache, send_fn=send_fn) if cache is not None else None
    kwargs = {"sleep": sleep} if sleep is not None else {}
    client = ReliableClient(registry, config.snapshot().routing, send_fn=cached or send_fn, **kwargs)
    log = log or InMemoryRequestLog()
    if hash_secret is None:
        hash_secret = hash_secret_from_env(required=isinstance(log, SQLiteRequestLog))
    from app.api.background import BackgroundLabeler
    from app.api.verification import check_background_budget
    judge_settings = judge_settings or load_judge_settings()
    deps = RouterDeps(registry, config, client, predictor, log, classifier, cached, fake_models, hash_secret,
                      checker_settings or load_checker_settings(registry=registry), judge_settings,
                      labeler or BackgroundLabeler())
    # refuse to start if background labels could eat more than half of the strong or judge model's daily budget
    check_background_budget(config.snapshot().verification.background, [deps.strong_model(), judge_settings.model_name])
    return deps


def load_predictor_checked():
    """The classifier named in routing.yaml, loaded and warmed once. Refuses to start if its checks fail."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")      # the embedding model must already be cached
    from app.router.predictor import Predictor, PredictorError, load_classifier_config
    name, version = load_classifier_config()
    try:
        predictor = Predictor(name, version)          # checks the version inside the file and feature columns
        predictor.predict_text("warm up the embedding model", None)
    except PredictorError as error:
        raise RuntimeError(f"refusing to start: the predictor failed its safety checks: {error}") from None
    return predictor, f"{name} v{version}"


def load_deps() -> RouterDeps:
    """Production (or TRIAGE_FAKE_MODELS=1) dependencies from config files and the environment."""
    fake = os.environ.get(FAKE_MODELS_ENV) == "1"
    if fake:
        from app.api.fake_models import FakeModels
        send_fn = FakeModels()
    else:
        send_fn = send_request
    log = open_request_log()
    hash_secret = hash_secret_from_env(required=isinstance(log, SQLiteRequestLog))   # before loading anything slow
    cache = None
    if os.environ.get(CACHE_ENV, "on").lower() != "off":
        cache = ResponseCache(os.environ.get(CACHE_PATH_ENV) or DEFAULT_CACHE_PATH)
    predictor, classifier = load_predictor_checked()
    return build_deps(predictor, send_fn, cache=cache, classifier=classifier, log=log,
                      fake_models=send_fn if fake else None, hash_secret=hash_secret)
