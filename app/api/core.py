"""The core of a chat request: decide the route, then stream the chosen model's answer (Phase 4 Step 5).

route_and_stream() is left for you to write; tests/test_api_core.py fails until then. The HTTP layer
(app/api/main.py) calls it for every /v1/chat/completions request, streaming or not, and turns what it
returns into OpenAI's response format, headers and the request log.

The helpers below (GenerationSettings, RouteDecision, RoutedStream, aiter_in_thread) are ready to use.
"""
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Optional

import anyio

from app.providers.base import Response


class UnknownModel(LookupError):
    """The request named a model that is neither "auto" nor in config/models.yaml."""

    def __init__(self, name: str):
        super().__init__(f"The model `{name}` does not exist. Use \"auto\" or a name from GET /v1/models.")
        self.name = name


@dataclass(frozen=True)
class GenerationSettings:
    """The caller's own generation settings; None = use routing.yaml's (the Phase 2 settings)."""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    def as_kwargs(self) -> dict:
        """Only the values the caller set, ready for ReliableClient.send(**...)."""
        return {k: v for k, v in (("temperature", self.temperature), ("max_tokens", self.max_tokens)) if v is not None}


@dataclass(frozen=True)
class RouteDecision:
    request_id: str
    requested_model: str            # "auto" or a pass-through registry name
    tier: str                       # low | medium | high (a pass-through model's own tier)
    primary_model: str              # first model in the tier's chain (or the pass-through model)
    model: str                      # the model actually answering (differs after a fallback)
    reason: str                     # app.router.quality.RoutingDecision.reason, or "requested model <name>"
    tau: Optional[float] = None     # None for pass-through
    p_low: Optional[float] = None
    p_medium: Optional[float] = None
    task_type: Optional[str] = None       # the task type the predictor used (given or guessed)
    task_type_guessed: Optional[bool] = None
    attempts: tuple = ()            # app.reliability.Attempt records from before the first token

    @property
    def fell_back(self) -> bool:
        return self.model != self.primary_model


class RoutedStream:
    """What route_and_stream returns: the decision, the answer's text pieces, and the full Response.

    pieces    an async iterator of text pieces; iterate it once
    response  None until pieces is exhausted, then the full Response (tokens, costs, cached, latency)
    """

    def __init__(self, decision: RouteDecision, pieces: AsyncIterator[str], response_fn: Callable[[], Optional[Response]]):
        self.decision = decision
        self.pieces = pieces
        self._response_fn = response_fn

    @property
    def response(self) -> Optional[Response]:
        return self._response_fn()

    @classmethod
    def from_stream(cls, decision: RouteDecision, stream) -> "RoutedStream":
        """Wrap a ReliableResult.response from send(stream=True): its pieces are pulled in a worker thread;
        its .response appears once they run out."""
        return cls(decision, aiter_in_thread(stream), lambda: stream.response)

    @classmethod
    def from_response(cls, decision: RouteDecision, response: Response) -> "RoutedStream":
        """Wrap a finished (non-streaming) Response: one piece with the whole text; .response once it's read."""
        done = {}

        async def one_piece():
            yield response.text
            done["response"] = response

        return cls(decision, one_piece(), lambda: done.get("response"))


_END = object()


async def aiter_in_thread(iterable) -> AsyncIterator:
    """Iterate a blocking (sync) iterable from async code, one next() per worker-thread call."""
    iterator = iter(iterable)
    while True:
        item = await anyio.to_thread.run_sync(next, iterator, _END)
        if item is _END:
            return
        yield item


async def route_and_stream(messages: list, *, tau: Optional[float], task_type: Optional[str], model: str = "auto",
                           stream: bool = True, settings: GenerationSettings, deps, request_id: str) -> RoutedStream:
    """Decide the route for one chat request, start the chosen model, and return once its answer has begun.

    messages    OpenAI-style [{"role": ..., "content": ...}], already validated
    tau         the quality dial for model="auto" (X-Triage-Quality, already resolved); ignored for pass-through
    task_type   a normalized task type (X-Triage-Task) or None (the predictor guesses it); auto only
    model       "auto", or a model name from deps.registry (pass-through)
    stream      True: stream the answer. False: one non-streaming call (it goes through the response cache)
    settings    the caller's temperature / max_tokens (None = routing.yaml's); pass settings.as_kwargs() on
    deps        app/api/deps.RouterDeps: .predictor, .client (ReliableClient), .registry
    request_id  goes into the RouteDecision unchanged

    model="auto":
      1. prediction = deps.predictor.predict_messages(messages, task_type)
      2. tier = app.router.quality.choose_tier(prediction.p_low, prediction.p_medium, tau)
         (reason: app.router.quality.route(...).reason)
      3. deps.client.send(messages, tier, stream=stream, **settings.as_kwargs()), which walks the tier's
         fallback chain
    Pass-through (a registry name): no prediction; deps.client.send_model(messages, model, stream=stream, ...),
      one model with its retries and timeouts but no fallback to another model. tau and p_* are None.
    Anything else: raise UnknownModel(model) before calling anything.

    ReliableClient is blocking: call it in a worker thread (anyio.to_thread.run_sync) so the event loop keeps
    serving other requests. send(stream=True) returns only after the first piece has arrived, so:
      - fallback (including the streaming first-token limit) happens only BEFORE this function returns;
        if every model fails first, ReliableClient raises AllModelsFailed: let it propagate
      - once this returns, the route is decided: RouteDecision.model is the model that is answering
      - an error while iterating .pieces after that propagates to the caller (no fallback mid-answer)

    Returns a RoutedStream (use RoutedStream.from_stream / .from_response):
      .decision  a RouteDecision (model = the ReliableResult's model.name, primary_model = its .primary,
                 attempts = tuple(its .attempts))
      .pieces    async iterator of text pieces (stream=False: the whole text as one piece)
      .response  None until .pieces is exhausted, then the full Response
    """
    raise NotImplementedError("route_and_stream: write this one yourself")
