"""Shared fakes for the API tests (Phase 4 Step 5). No network, no API keys, no real models.

reference_route_and_stream is a reference version of app/api/core.route_and_stream, so the HTTP tests in
test_api.py test the API layer around it. The real function has its own tests in test_api_core.py.
"""
import copy
import time

import anyio
import yaml

from app.api import core
from app.api.config_state import ActiveConfig
from app.api.deps import build_deps
from app.api.fake_models import FakeModels
from app.reliability import DEFAULT_ROUTING_PATH
from app.router.predictor import Prediction
from app.router.quality import route

API_KEY, OTHER_KEY, ADMIN_KEY = "test-key-123", "other-key-456", "admin-key-789"
TEST_HASH_SECRET = b"fixed-test-hash-secret"
ASK = [{"role": "user", "content": "What is 2 + 3? End with: Answer: <number>"}]


class FakePredictor:
    def __init__(self, p_low=0.9, p_medium=0.95, guessed_task="qa"):
        self.p_low, self.p_medium, self.guessed_task = p_low, p_medium, guessed_task
        self.calls = []

    def predict_messages(self, messages, task_type=None):
        self.calls.append((messages, task_type))
        return Prediction(p_low=self.p_low, p_medium=max(self.p_medium, self.p_low), raw_p_medium=self.p_medium,
                          task_type=task_type or self.guessed_task, task_type_guessed=task_type is None)


def routing_raw(first_token=None) -> dict:
    raw = yaml.safe_load(DEFAULT_ROUTING_PATH.read_text(encoding="utf-8"))
    raw = copy.deepcopy(raw)
    if first_token is not None:
        raw["streaming"] = {"first_token_timeout_seconds": first_token}
    return raw


def make_deps(script=None, *, predictor=None, first_token=None, cache=None, model_sleep=time.sleep, log=None,
              hash_secret=TEST_HASH_SECRET):
    """Real registry, real routing.yaml (optionally a short first-token limit), real ReliableClient
    (backoff sleeps skipped), fake models."""
    from app.registry import load_registry
    registry = load_registry()
    fake = FakeModels(script, sleep=model_sleep)
    deps = build_deps(predictor or FakePredictor(), fake, registry=registry,
                      config=ActiveConfig(routing_raw(first_token), registry), cache=cache,
                      classifier="classifier_c_logreg_cv v2", sleep=lambda s: None, fake_models=fake, log=log,
                      hash_secret=hash_secret)
    return deps


async def reference_route_and_stream(messages, *, tau, task_type, model="auto", stream=True, settings, deps,
                                     request_id):
    kwargs = settings.as_kwargs()
    if model == "auto":
        p = deps.predictor.predict_messages(messages, task_type)
        rd = route(p.p_low, p.p_medium, tau)
        result = await anyio.to_thread.run_sync(lambda: deps.client.send(messages, rd.tier, stream=stream, **kwargs))
        extra = dict(reason=rd.reason, tau=tau, p_low=p.p_low, p_medium=p.p_medium, task_type=p.task_type,
                     task_type_guessed=p.task_type_guessed)
    elif model in deps.registry:
        result = await anyio.to_thread.run_sync(lambda: deps.client.send_model(messages, model, stream=stream, **kwargs))
        extra = dict(reason=f"requested model {model}")
    else:
        raise core.UnknownModel(model)
    decision = core.RouteDecision(request_id=request_id, requested_model=model, tier=result.tier,
                                  primary_model=result.primary, model=result.model.name,
                                  attempts=tuple(result.attempts), **extra)
    if stream:
        return core.RoutedStream.from_stream(decision, result.response)
    return core.RoutedStream.from_response(decision, result.response)
