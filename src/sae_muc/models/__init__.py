"""LLM and NLI backends — common interfaces across HF-local, OpenRouter, and fakes."""

from sae_muc.models.base import Generation, LLMBackend
from sae_muc.models.fake import FakeBackend
from sae_muc.models.hf_local import HFLocalBackend
from sae_muc.models.nli import (
    FakeNLIBackend,
    HFLocalNLIBackend,
    NLIBackend,
    build_nli_backend,
)
from sae_muc.models.openrouter import MissingAPIKeyError, OpenRouterBackend
from sae_muc.models.registry import build_llm_backend

__all__ = [
    "FakeBackend",
    "FakeNLIBackend",
    "Generation",
    "HFLocalBackend",
    "HFLocalNLIBackend",
    "LLMBackend",
    "MissingAPIKeyError",
    "NLIBackend",
    "OpenRouterBackend",
    "build_llm_backend",
    "build_nli_backend",
]
