"""LLM backends — one common interface for HF-local, OpenRouter, and fake providers."""

from sae_muc.models.base import Generation, LLMBackend
from sae_muc.models.fake import FakeBackend
from sae_muc.models.hf_local import HFLocalBackend
from sae_muc.models.openrouter import MissingAPIKeyError, OpenRouterBackend
from sae_muc.models.registry import build_llm_backend

__all__ = [
    "FakeBackend",
    "Generation",
    "HFLocalBackend",
    "LLMBackend",
    "MissingAPIKeyError",
    "OpenRouterBackend",
    "build_llm_backend",
]
