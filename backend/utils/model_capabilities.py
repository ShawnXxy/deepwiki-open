"""Model capability helpers.

The Azure OpenAI ``gpt-5`` / ``o*`` reasoning families reject a different
set of parameters than chat models. The implementation lives in
``backend.model_routing`` so configured deployment aliases can be kept
separate from their underlying model names.
"""

from backend.model_routing import is_reasoning_model

__all__ = ["is_reasoning_model"]
