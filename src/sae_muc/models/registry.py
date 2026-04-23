"""Single entry point that turns a (provider, model, options) triple into an LLMBackend."""

from __future__ import annotations

from sae_muc.models.base import LLMBackend
from sae_muc.models.fake import FakeBackend
from sae_muc.models.hf_local import HFLocalBackend
from sae_muc.models.openrouter import OpenRouterBackend


def build_llm_backend(
    provider: str,
    model: str,
    *,
    dtype: str = "bfloat16",
    max_retries: int = 3,
) -> LLMBackend:
    if provider == "fake":
        return FakeBackend(name=model)
    if provider == "openrouter":
        return OpenRouterBackend(model=model, max_retries=max_retries)
    if provider == "hf_local":
        return HFLocalBackend(model_name=model, dtype=dtype)
    raise ValueError(f"Unknown backend provider: {provider!r}")
