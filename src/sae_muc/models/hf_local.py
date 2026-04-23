"""HuggingFace-local LLM backend (stub).

The real implementation — loading the model with transformers, running
`.generate` in batches, extracting residual-stream hidden states, and
attaching forward hooks for MUC intervention — lands with the
`hidden_states` pipeline stage. This stub exists so the registry can
dispatch on `provider='hf_local'` today and the pipeline fails with a
clear message if the backend is invoked before the real code is wired up.
"""

from __future__ import annotations

from sae_muc.models.base import Generation


class HFLocalBackend:
    def __init__(self, model_name: str, *, dtype: str = "bfloat16") -> None:
        self.name = model_name
        self._dtype = dtype

    def generate(
        self,
        prompts: list[str],
        *,
        temperature: float,
        max_new_tokens: int,
        n: int = 1,
        system: str | None = None,
    ) -> list[list[Generation]]:
        raise NotImplementedError(
            "HFLocalBackend.generate is not implemented yet. "
            "It lands with the hidden_states pipeline stage."
        )
