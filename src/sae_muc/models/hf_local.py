"""HuggingFace-local LLM backend.

Real implementation of `generate`, `hidden_states`, and helpers. Heavy
imports (`torch`, `transformers`) are done inside `__init__` so callers
that use only the Fake or OpenRouter backends do not pay their cost.

Performance optimisations (FlashAttention-2, batched generation,
num_return_sequences, bnb quantisation, pre-downloaded weights, etc.)
are deliberately out of scope — see `TODO.md`. Generation processes
prompts in a small batch with left-padding; hidden-state extraction runs
one prompt at a time to avoid padding bookkeeping. Both are correct and
good enough for MVP.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sae_muc.models.base import Generation

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

_DTYPE_MAP: dict[str, str] = {
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
}


class HFLocalBackend:
    def __init__(self, model_name: str, *, dtype: str = "bfloat16") -> None:
        # Lazy load: `__init__` is cheap, the actual model/tokeniser are
        # fetched on first use. This keeps build_context() cheap for stages
        # that don't touch the model (e.g. CLI `--help`, registry tests).
        self.name = model_name
        self._dtype = dtype
        self._model = None
        self._tokenizer = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, _DTYPE_MAP[self._dtype])
        has_cuda = torch.cuda.is_available()
        device_map = "auto" if has_cuda else None
        log.info("Loading %s (dtype=%s, cuda=%s)", self.name, self._dtype, has_cuda)

        tokenizer = AutoTokenizer.from_pretrained(self.name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._device = next(model.parameters()).device

    # ----------------------------------------------------------------- #
    # Generation                                                         #
    # ----------------------------------------------------------------- #

    def generate(
        self,
        prompts: list[str],
        *,
        temperature: float,
        max_new_tokens: int,
        n: int = 1,
        system: str | None = None,
    ) -> list[list[Generation]]:
        import torch

        self._ensure_loaded()
        texts = [self._apply_chat_template(p, system=system) for p in prompts]
        prev_side = self._tokenizer.padding_side
        self._tokenizer.padding_side = "left"
        try:
            inputs = self._tokenizer(texts, return_tensors="pt", padding=True).to(self._device)
        finally:
            self._tokenizer.padding_side = prev_side

        input_len = inputs.input_ids.shape[1]
        do_sample = temperature > 1e-5
        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": n,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            out = self._model.generate(**inputs, **gen_kwargs)

        decoded = self._tokenizer.batch_decode(out[:, input_len:], skip_special_tokens=True)

        result: list[list[Generation]] = []
        for i in range(len(prompts)):
            result.append(
                [
                    Generation(text=decoded[i * n + j], finish_reason="stop")
                    for j in range(n)
                ]
            )
        return result

    def _apply_chat_template(self, prompt: str, *, system: str | None) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        if self._tokenizer.chat_template:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # No chat template — fall back to the raw prompt (useful for base models / tests).
        return prompt

    # ----------------------------------------------------------------- #
    # Hidden states                                                      #
    # ----------------------------------------------------------------- #

    def hidden_states(self, texts: list[str]) -> list["torch.Tensor"]:
        """Forward each text once; return [n_layers+1, seq_len, d_model] tensors.

        Index 0 is the token-embedding output; indices 1..n_layers are
        transformer-block residual-stream outputs. Tensors are returned on CPU
        in float32 so downstream code (safetensors, analysis) can consume them
        uniformly regardless of the model's compute dtype.
        """
        import torch

        self._ensure_loaded()
        result: list[torch.Tensor] = []
        for text in texts:
            inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
            with torch.inference_mode():
                out = self._model(**inputs, output_hidden_states=True, return_dict=True)
            # tuple of (n_layers+1) tensors, each shape [1, seq_len, d_model]
            stacked = torch.stack(out.hidden_states, dim=0).squeeze(1)
            result.append(stacked.float().cpu())
        return result

    def tokenize_length(self, text: str, *, add_special_tokens: bool = True) -> int:
        self._ensure_loaded()
        return len(self._tokenizer(text, add_special_tokens=add_special_tokens).input_ids)

    # ----------------------------------------------------------------- #
    # Generation with forward hook (for MUC intervention)                #
    # ----------------------------------------------------------------- #

    def generate_with_hook(
        self,
        prompts: list[str],
        *,
        hook_layer: int,
        hook_fn,
        temperature: float,
        max_new_tokens: int,
        n: int = 1,
        system: str | None = None,
    ) -> list[list[Generation]]:
        """Same as `generate` but with `hook_fn` applied at `hook_layer`'s residual stream.

        `hook_fn(residual: [B, T, D]) -> [B, T, D]` runs after every forward of the
        target transformer block. Path is Llama/Mistral/Qwen2-compatible
        (`model.model.layers[i]`); other architectures will need a layer-map update.
        """
        import torch

        self._ensure_loaded()
        texts = [self._apply_chat_template(p, system=system) for p in prompts]
        prev_side = self._tokenizer.padding_side
        self._tokenizer.padding_side = "left"
        try:
            inputs = self._tokenizer(texts, return_tensors="pt", padding=True).to(self._device)
        finally:
            self._tokenizer.padding_side = prev_side

        input_len = inputs.input_ids.shape[1]
        do_sample = temperature > 1e-5
        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": n,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        def _wrapped(module, _inputs, output):
            if isinstance(output, tuple):
                return (hook_fn(output[0]),) + output[1:]
            return hook_fn(output)

        target = self._model.model.layers[hook_layer]
        handle = target.register_forward_hook(_wrapped)
        try:
            with torch.inference_mode():
                out = self._model.generate(**inputs, **gen_kwargs)
        finally:
            handle.remove()

        decoded = self._tokenizer.batch_decode(out[:, input_len:], skip_special_tokens=True)
        result: list[list[Generation]] = []
        for i in range(len(prompts)):
            result.append(
                [
                    Generation(text=decoded[i * n + j], finish_reason="stop")
                    for j in range(n)
                ]
            )
        return result
