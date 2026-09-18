"""LLM access: model fallback, retries, budgets and an offline stub.

Everything the agent sends to a model goes through an LLMProvider. Two exist:

  GeminiProvider  the real thing, over google-genai.
  StubProvider    deterministic, scripted, no network.

The stub is not only a test fixture. It is how the prototype stays demonstrable
when the API key has no quota — which is exactly the state this project's key
was in on 2026-09-18, when every model returned
"429 — Your prepayment credits are depleted". Requirement 5 asks for resilience
to third-party downtime; a hard dependency on a vendor being up would fail that
requirement in the demo itself.

Failure handling is deliberately asymmetric, because retrying the wrong error is
how an agent turns an outage into a bill:

  5xx / UNAVAILABLE      transient  -> retry with exponential backoff + jitter
  429 rate limited       transient  -> retry, then fall back to the smaller model
  429 quota exhausted    FATAL      -> do not retry. More requests cannot succeed;
                                       only a human topping up credits can fix it.
  404 model not found    FATAL for  -> fall back to the configured secondary model
                         that model    (gemini-2.5-* are retired for new keys, and
                                       the API returns 404 with that advice)
  401 / 403              FATAL      -> a misconfigured key; surface to the operator

A per-turn call budget caps the blast radius of a self-correction loop that will
not converge.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, Sequence

DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_FALLBACK_MODEL = "gemini-3.1-flash-lite"

# Hard ceiling on model calls in one user turn. A turn normally uses 2-4: plan,
# maybe a self-correction, then compose. Anything past this is a loop.
DEFAULT_CALL_BUDGET = 8


class LLMError(Exception):
    """Base for model failures."""

    retryable = False


class LLMTransientError(LLMError):
    retryable = True


class LLMQuotaError(LLMError):
    """Quota or credits exhausted — retrying cannot help."""


class LLMConfigError(LLMError):
    """Bad key, bad project, or a malformed request. Fatal — no fallback.

    Sending the same malformed request to a second model just doubles the cost
    of failing.
    """


class LLMModelUnavailableError(LLMConfigError):
    """THIS model cannot serve the request, but another one might.

    The case that matters: Gemini retires models for new keys and answers 404
    with "no longer available". Falling back to the configured secondary is the
    correct response — unlike a 400, where the request itself is wrong.
    """


class LLMBudgetError(LLMError):
    """The turn exceeded its allowed number of model calls."""


@dataclass(frozen=True)
class FunctionCall:
    name: str
    args: dict[str, Any]
    # Gemini 3.x returns an opaque thought_signature alongside each function
    # call, and REQUIRES it to be echoed back verbatim when that call appears in
    # conversation history. Drop it and the next request fails with
    # "400 Function call is missing a thought_signature in functionCall parts",
    # which means every tool-using turn dies on its second model call.
    thought_signature: Any = None


@dataclass(frozen=True)
class LLMResponse:
    text: str = ""
    function_calls: tuple[FunctionCall, ...] = ()
    prompt_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    used_fallback: bool = False

    @property
    def wants_tool(self) -> bool:
        return bool(self.function_calls)


class LLMProvider(Protocol):
    """What the agent depends on. Keeps the orchestrator testable offline."""

    def generate(
        self,
        *,
        system: str,
        contents: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]] = (),
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


def _classify_genai(err: Exception) -> LLMError:
    """Map a google-genai exception to a retry policy."""
    from google.genai import errors as genai_errors

    message = str(err)
    lowered = message.lower()
    code = getattr(err, "code", None)

    if isinstance(err, genai_errors.ServerError) or code in {500, 502, 503, 504}:
        return LLMTransientError(message)

    if code == 429 or "resource_exhausted" in lowered or "rate limit" in lowered:
        # Distinguish "slow down" from "you are out of money". Only the first is
        # worth retrying, and confusing them produces a loop that never succeeds.
        if any(
            token in lowered
            for token in ("credits are depleted", "quota exceeded", "billing",
                          "exhausted your current quota", "insufficient")
        ):
            return LLMQuotaError(
                f"{message}\nThe API key has no remaining quota — retrying will not "
                "help. Top up credits in AI Studio or use a key with free-tier quota."
            )
        return LLMTransientError(message)

    if code == 400 or "invalid_argument" in lowered:
        # A malformed request is our bug, not the service's. Retrying it spends
        # the turn's entire call budget on an error that cannot change — which
        # is exactly how a self-correction loop turns into a cost incident.
        return LLMConfigError(message)

    if code in {401, 403} or "api key not valid" in lowered or "permission" in lowered:
        return LLMConfigError(message)

    if code == 404 or "not found" in lowered or "no longer available" in lowered:
        return LLMModelUnavailableError(message)

    if isinstance(err, genai_errors.APIError):
        return LLMTransientError(message)

    return LLMError(message)


@dataclass
class GeminiProvider:
    """google-genai backed provider with model fallback and retries.

    Two backends, same wire format:

      AI Studio  an API key. Simplest to obtain, but the free tier is small and
                 a depleted key returns 429 on every model.
      Vertex AI  Application Default Credentials against a GCP project. No API
                 key at all, billed through the project, and it reuses the same
                 credentials BigQuery already needs — so a deployment that can
                 read the warehouse can also reach the model.

    Vertex is the better production answer regardless: keys do not have to be
    minted, stored or rotated, and access is IAM rather than a bearer secret.
    """

    api_key: str | None = None
    use_vertex: bool = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "").lower() == "vertex"
    )
    project: str | None = field(
        default_factory=lambda: os.getenv("GOOGLE_CLOUD_PROJECT")
    )
    location: str = field(
        default_factory=lambda: os.getenv("VERTEX_LOCATION", "global")
    )
    model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", DEFAULT_MODEL))
    fallback_model: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL
        )
    )
    temperature: float = 0.2
    max_attempts: int = 3
    call_budget: int = DEFAULT_CALL_BUDGET
    client: Any | None = None
    sleep: Callable[[float], None] = time.sleep

    _calls_this_turn: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.client is not None:
            return

        from google import genai

        if self.use_vertex:
            if not self.project:
                raise LLMConfigError(
                    "LLM_PROVIDER=vertex needs GOOGLE_CLOUD_PROJECT set, and "
                    "Application Default Credentials "
                    "(`gcloud auth application-default login`)."
                )
            try:
                self.client = genai.Client(
                    vertexai=True, project=self.project, location=self.location
                )
            except Exception as err:  # noqa: BLE001
                raise LLMConfigError(
                    f"could not reach Vertex AI in project {self.project!r}: {err}. "
                    "Enable it with `gcloud services enable aiplatform.googleapis.com`."
                ) from err
            return

        key = self.api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise LLMConfigError(
                "GOOGLE_API_KEY is not set. Copy .env.example to .env and add a "
                "Google AI Studio key, set LLM_PROVIDER=vertex to use Application "
                "Default Credentials instead, or LLM_PROVIDER=stub to run offline."
            )
        self.client = genai.Client(api_key=key)

    def begin_turn(self) -> None:
        """Reset the per-turn call budget."""
        self._calls_this_turn = 0

    def generate(
        self,
        *,
        system: str,
        contents: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]] = (),
    ) -> LLMResponse:
        self._calls_this_turn += 1
        if self._calls_this_turn > self.call_budget:
            raise LLMBudgetError(
                f"this turn used its budget of {self.call_budget} model calls "
                "without reaching an answer"
            )

        models = [self.model]
        if self.fallback_model and self.fallback_model != self.model:
            models.append(self.fallback_model)

        last: LLMError | None = None
        for index, model in enumerate(models):
            try:
                return self._call_with_retries(
                    model=model,
                    system=system,
                    contents=contents,
                    tools=tools,
                    used_fallback=index > 0,
                )
            except LLMModelUnavailableError as err:
                # This model is retired or unavailable: try the next one rather
                # than failing the turn. Note this does NOT catch plain
                # LLMConfigError — a malformed request or a bad key fails the
                # same way on every model, so falling back would only pay twice.
                last = err
                continue
            except LLMQuotaError:
                # Every model on the key shares the quota pool — trying another
                # is pointless.
                raise
        raise last or LLMConfigError("no usable model configured")

    def _call_with_retries(
        self,
        *,
        model: str,
        system: str,
        contents: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        used_fallback: bool,
    ) -> LLMResponse:
        from google.genai import types

        config: dict[str, Any] = {
            "system_instruction": system,
            "temperature": self.temperature,
        }
        if tools:
            config["tools"] = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(**declaration) for declaration in tools
                    ]
                )
            ]
            # The agent drives the tool loop itself: every call is guarded,
            # traced and budgeted, which automatic execution would bypass.
            config["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
                disable=True
            )

        last: LLMError | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(**config),
                )
            except Exception as err:  # noqa: BLE001
                error = _classify_genai(err)
                if not error.retryable:
                    raise error from err
                last = error
                if attempt < self.max_attempts:
                    self.sleep(min(2 ** (attempt - 1), 8) * (0.5 + random.random()))
                continue
            return _parse_response(response, model=model, used_fallback=used_fallback)

        raise last or LLMTransientError("model call failed")


def _parse_response(response: Any, *, model: str, used_fallback: bool) -> LLMResponse:
    """Pull text, tool calls and token usage out of a genai response."""
    text_parts: list[str] = []
    calls: list[FunctionCall] = []

    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "text", None):
                text_parts.append(part.text)
            call = getattr(part, "function_call", None)
            if call is not None:
                calls.append(
                    FunctionCall(
                        name=call.name,
                        args=dict(call.args or {}),
                        # Carried on the PART, not the call, and opaque to us —
                        # it must survive the round trip untouched.
                        thought_signature=getattr(part, "thought_signature", None),
                    )
                )

    usage = getattr(response, "usage_metadata", None)
    return LLMResponse(
        text="".join(text_parts).strip(),
        function_calls=tuple(calls),
        prompt_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
        output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
        model=model,
        used_fallback=used_fallback,
    )


# ---------------------------------------------------------------------------
# Offline stub
# ---------------------------------------------------------------------------


@dataclass
class StubProvider:
    """Deterministic provider for tests and zero-quota demos.

    Scripted with a queue of LLMResponse objects, or with a callable that picks a
    response from the conversation so far. Records every request for assertions.
    """

    script: list[LLMResponse] = field(default_factory=list)
    responder: Callable[[str, list[dict[str, Any]]], LLMResponse] | None = None
    call_budget: int = DEFAULT_CALL_BUDGET

    requests: list[dict[str, Any]] = field(default_factory=list, init=False)
    _calls_this_turn: int = field(default=0, init=False)

    def begin_turn(self) -> None:
        self._calls_this_turn = 0

    def generate(
        self,
        *,
        system: str,
        contents: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]] = (),
    ) -> LLMResponse:
        self._calls_this_turn += 1
        if self._calls_this_turn > self.call_budget:
            raise LLMBudgetError(
                f"this turn used its budget of {self.call_budget} model calls"
            )
        self.requests.append({"system": system, "contents": contents, "tools": tools})

        if self.responder is not None:
            return self.responder(system, contents)
        if self.script:
            return self.script.pop(0)
        return LLMResponse(text="(stub: no scripted response left)", model="stub")


def build_provider(**kwargs: Any) -> LLMProvider:
    """Pick a provider from the environment.

    LLM_PROVIDER:
      stub    no network, no quota — the whole agent runs offline
      vertex  Vertex AI via Application Default Credentials (no API key)
      gemini  AI Studio API key (default)
    """
    choice = os.getenv("LLM_PROVIDER", "gemini").lower()
    if choice == "stub":
        return StubProvider(**kwargs)
    if choice == "vertex":
        kwargs.setdefault("use_vertex", True)
    return GeminiProvider(**kwargs)
