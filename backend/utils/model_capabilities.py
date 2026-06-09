"""Model capability helpers.

The Azure OpenAI ``gpt-5`` / ``o*`` reasoning families reject a different
set of parameters than the chat models (no ``temperature``, no
``top_p`` -- but they accept ``reasoning_effort``, ``verbosity`` and
``max_completion_tokens``). Callers branch on these helpers instead of
hardcoding deployment names so swapping a deployment in
``infra.json`` does not require code changes.
"""

from __future__ import annotations

# Lowercase prefixes that identify a reasoning-family deployment.
# Matched against the leading characters of the deployment name. New
# reasoning families (o5, o6, gpt-6 ...) only need to be appended here.
_REASONING_PREFIXES = (
    "gpt-5",
    "o1",
    "o3",
    "o4",
)


def is_reasoning_model(deployment_name: str | None) -> bool:
    """Return True if the deployment is a reasoning-family model.

    Defaults to False on falsy input so callers fall through to the
    chat-style parameter set when a deployment is unknown / empty.

    A ``-chat`` suffix flips the classification back to chat-tier
    (Azure ships ``gpt-5.1-chat`` as a non-reasoning deployment of the
    gpt-5 family -- it accepts ``temperature`` and rejects
    ``reasoning_effort``).
    """
    if not deployment_name:
        return False
    name = deployment_name.strip().lower()
    if name.endswith("-chat"):
        return False
    return any(name.startswith(p) for p in _REASONING_PREFIXES)
