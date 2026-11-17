"""Everything a request needs, built once at startup: predictor, ReliableClient, config, cache, log."""
import os
from dataclasses import dataclass, field
from typing import Optional

from app.api.config_state import ActiveConfig
from app.api.request_log import InMemoryRequestLog, SQLRequestLog, hash_secret_from_env, open_request_log
from pathlib import Path

from app.api.demo import DemoLimiter, DemoSettings, load_demo_settings, trusted_hops_from_env
from app.cache import DEFAULT_CACHE_PATH, FAKE_CACHE_PATH, CachedSender, ResponseCache
from app.providers.base import send_request
from app.registry import load_registry
from app.reliability import ReliableClient
from app.verifier.ai_checker import CheckerSettings, load_checker_settings
from app.verifier.judge import JudgeSettings, load_judge_settings

FAKE_MODELS_ENV, CACHE_ENV, CACHE_PATH_ENV = "TRIAGE_FAKE_MODELS", "TRIAGE_CACHE", "TRIAGE_CACHE_PATH"
SERVER_PROVIDERS_ENV = "TRIAGE_SERVER_PROVIDERS"     # e.g. "groq,gemini" on a host that can't run Ollama


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
    demo_settings: Optional[DemoSettings] = None         # config/demo.yaml
    demo_limiter: Optional[DemoLimiter] = None           # demo and public-BYOK counters (in the database)
    trusted_proxy_hops: int = 0                          # TRIAGE_TRUSTED_PROXY_HOPS, for the visitor's IP
    engine: Optional[object] = None                      # the database (app/db.py), shared by everything below
    accounts: Optional[object] = None                    # app.accounts.store.AccountStore
    auth_limiter: Optional[object] = None                # app.accounts.ratelimit.AuthRateLimiter
    keyring: Optional[object] = field(default=None, repr=False)   # TRIAGE_ENCRYPTION_KEYS (accounts on)
    provider_keys: Optional[object] = None               # app.accounts.provider_keys.ProviderKeyStore
    key_verifier: Optional[object] = None                # app.accounts.verify_keys.verify_key (tests: a fake)
    api_keys: Optional[object] = None                    # app.accounts.api_keys.ApiKeyStore
    account_limiter: Optional[object] = None             # app.accounts.limits.AccountLimiter (account chat)
    password_resets: Optional[object] = None             # app.accounts.password_reset.ResetStore
    email_sender: Optional[object] = field(default=None, repr=False)   # app.accounts.email (chosen at startup)

    def strong_model(self, providers=None, prefer=None) -> str:
        """The savings baseline: the first model of the high chain after filtering by provider, and reordering
        for a BYOK request's preferred provider (prefer="anthropic": claude-opus).
        providers=None means routing's server_providers (requests without keys: groq-large)."""
        chain = self.config.snapshot().routing.chain_for("high", self.registry, providers, prefer)
        if not chain:
            raise LookupError("no model in the high tier is on an allowed provider")
        return chain[0]

    def update_config(self, patch: dict):
        snapshot = self.config.update(patch)          # raises ConfigUpdateError and changes nothing if invalid
        self.client.routing = snapshot.routing        # the breaker (and its memory of failures) is kept
        return snapshot


def build_deps(predictor, send_fn, *, registry=None, config: Optional[ActiveConfig] = None,
               cache: Optional[ResponseCache] = None, log=None, classifier: str = "", sleep=None,
               fake_models=None, hash_secret: Optional[bytes] = None, labeler=None,
               checker_settings: Optional[CheckerSettings] = None,
               judge_settings: Optional[JudgeSettings] = None, demo_settings: Optional[DemoSettings] = None,
               demo_limiter: Optional[DemoLimiter] = None, trusted_proxy_hops: int = 0,
               engine=None, keyring=None, key_verifier=None, account_limits=None) -> RouterDeps:
    """engine: the database; None = the demo limiter's if one is given, else a new in-memory one (tests)."""
    registry = registry or load_registry()
    config = config or ActiveConfig.from_file(registry)
    log = log or InMemoryRequestLog()
    if hash_secret is None:
        hash_secret = hash_secret_from_env(required=isinstance(log, SQLRequestLog))
    # hash_secret also fingerprints BYOK keys in cache keys (never the key itself)
    cached = CachedSender(cache, send_fn=send_fn, hash_secret=hash_secret) if cache is not None else None
    kwargs = {"sleep": sleep} if sleep is not None else {}
    client = ReliableClient(registry, config.snapshot().routing, send_fn=cached or send_fn, **kwargs)
    from app.api.background import BackgroundLabeler
    from app.api.verification import check_background_budget
    judge_settings = judge_settings or load_judge_settings()
    deps = RouterDeps(registry, config, client, predictor, log, classifier, cached, fake_models, hash_secret,
                      checker_settings or load_checker_settings(registry=registry), judge_settings,
                      labeler or BackgroundLabeler())
    deps.demo_settings = demo_settings or (demo_limiter.settings if demo_limiter else load_demo_settings())
    if engine is None:
        from app import db
        engine = demo_limiter.engine if demo_limiter is not None else db.open_engine("sqlite://")
    deps.engine = engine
    deps.demo_limiter = demo_limiter or DemoLimiter(deps.demo_settings, engine=engine)
    deps.trusted_proxy_hops = trusted_proxy_hops
    from app.accounts.ratelimit import AuthRateLimiter
    from app.accounts.store import AccountStore
    deps.accounts, deps.auth_limiter = AccountStore(engine), AuthRateLimiter(engine)
    from app.accounts.provider_keys import ProviderKeyStore
    from app.accounts.verify_keys import verify_key
    deps.keyring = keyring
    deps.provider_keys = ProviderKeyStore(engine, keyring)
    deps.key_verifier = key_verifier or verify_key
    from app.accounts.api_keys import ApiKeyStore
    deps.api_keys = ApiKeyStore(engine, hash_secret)            # keys are kept as an HMAC with this secret
    from app.accounts.password_reset import ResetStore
    deps.password_resets = ResetStore(engine, hash_secret)      # codes are kept as an HMAC with this secret
    from app.accounts.limits import AccountLimiter, load_account_limits
    deps.account_limiter = AccountLimiter(account_limits or load_account_limits(), engine)
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


def cache_path(fake: bool):
    """The response cache file, or None when TRIAGE_CACHE=off. Fake-model mode gets its own file
    (FAKE_CACHE_PATH): real and fake answers are cached under the same keys (model name + prompt + settings), so
    sharing one file would let a fake answer be served in real mode, or the reverse. Each mode refuses the other's
    file even when TRIAGE_CACHE_PATH names it explicitly."""
    if os.environ.get(CACHE_ENV, "on").lower() == "off":
        return None
    path = Path(os.environ.get(CACHE_PATH_ENV) or (FAKE_CACHE_PATH if fake else DEFAULT_CACHE_PATH))
    forbidden = DEFAULT_CACHE_PATH if fake else FAKE_CACHE_PATH
    if path.resolve() == forbidden.resolve():
        raise RuntimeError(f"refusing to start: {'fake' if fake else 'real'}-model mode can't use the "
                           f"{forbidden.name} cache ({CACHE_PATH_ENV}); fake and real answers must be cached apart")
    return path


def server_providers_from_env() -> Optional[list]:
    """TRIAGE_SERVER_PROVIDERS as a list (it replaces routing.yaml's server_providers), or None when unset."""
    raw = os.environ.get(SERVER_PROVIDERS_ENV, "")
    names = [p.strip() for p in raw.split(",") if p.strip()]
    return names or None


def load_config(registry: dict) -> ActiveConfig:
    """routing.yaml, with TRIAGE_SERVER_PROVIDERS applied. Refuses to start on a bad override."""
    override = server_providers_from_env()
    try:
        return ActiveConfig.from_file(registry, server_providers=override)
    except ValueError as error:
        if override is None:
            raise
        raise RuntimeError(f"refusing to start: {SERVER_PROVIDERS_ENV}={','.join(override)}: {error}") from None


def load_deps() -> RouterDeps:
    """Production (or TRIAGE_FAKE_MODELS=1) dependencies from config files and the environment."""
    fake = os.environ.get(FAKE_MODELS_ENV) == "1"
    if fake:
        from app.api.fake_models import FakeModels
        send_fn = FakeModels()
    else:
        send_fn = send_request
    from app.accounts.verify_keys import fake_verify_key       # fake mode: no API calls, keys included
    # One database for the request log and the demo counters: DATABASE_URL, or fake mode's own file
    from app import db
    engine = db.open_engine(db.database_url(fake))
    log = open_request_log(fake=fake, engine=engine)
    hash_secret = hash_secret_from_env(required=isinstance(log, SQLRequestLog))   # before loading anything slow
    path = cache_path(fake)
    cache = ResponseCache(path) if path else None
    registry = load_registry()
    config = load_config(registry)                    # before the slow predictor, so a bad override fails fast
    demo_settings = load_demo_settings()
    demo_limiter = DemoLimiter(demo_settings, engine=engine)   # survives restarts and redeploys
    trusted_proxy_hops = trusted_hops_from_env()
    predictor, classifier = load_predictor_checked()
    return build_deps(predictor, send_fn, registry=registry, config=config, cache=cache, classifier=classifier,
                      log=log, fake_models=send_fn if fake else None, hash_secret=hash_secret,
                      demo_settings=demo_settings, demo_limiter=demo_limiter, trusted_proxy_hops=trusted_proxy_hops,
                      engine=engine, key_verifier=fake_verify_key if fake else None)
