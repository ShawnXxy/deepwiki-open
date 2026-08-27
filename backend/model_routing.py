"""Azure OpenAI model routing and request parameter helpers."""

from __future__ import annotations

from typing import Literal, Optional


ModelTask = Literal["chat", "reasoning", "premium_reasoning"]

_REASONING_PREFIXES = (
    "gpt-5",
    "o1",
    "o3",
    "o4",
)


def is_reasoning_model(deployment_name: str | None) -> bool:
    """Return whether a deployment uses reasoning-model parameters."""
    if not deployment_name:
        return False
    name = deployment_name.strip().lower()
    if name.endswith("-chat"):
        return False
    return any(name.startswith(prefix) for prefix in _REASONING_PREFIXES)


def select_chat_model_task(
    *,
    is_deep_research: bool,
    wiki_structure_request: bool = False,
    wiki_page_request: bool = False,
) -> ModelTask:
    """Select the configured deployment tier for a chat endpoint request."""
    if is_deep_research or wiki_structure_request:
        return "premium_reasoning"
    if wiki_page_request:
        return "reasoning"
    return "chat"


def build_model_kwargs(
    deployment: str,
    *,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
    max_completion_tokens: Optional[int] = None,
    stream: Optional[bool] = None,
) -> dict[str, object]:
    """Build parameters accepted by the selected model family."""
    kwargs: dict[str, object] = {"model": deployment}

    capability_name = model_name or deployment
    if is_reasoning_model(capability_name):
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if verbosity is not None:
            kwargs["verbosity"] = verbosity
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens
    elif temperature is not None:
        kwargs["temperature"] = temperature

    if stream is not None:
        kwargs["stream"] = stream

    return kwargs
