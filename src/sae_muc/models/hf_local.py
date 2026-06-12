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
    from collections.abc import Iterator

    import torch

log = logging.getLogger(__name__)

_DTYPE_MAP: dict[str, str] = {
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
}


def _add_truncation_kwargs(
    gen_kwargs: dict, top_p: float | None, top_k: int | None
) -> None:
    """Pin nucleus / top-K truncation for sampled decoding (paper App C).

    Only called on the sampling path (do_sample=True). A `None` value defers
    to the model's bundled generation_config, matching the pre-fix behaviour.
    """
    if top_p is not None:
        gen_kwargs["top_p"] = float(top_p)
    if top_k is not None:
        gen_kwargs["top_k"] = int(top_k)


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
        log.info("Loading %s (dtype=%s, cuda=%s)", self.name, self._dtype, has_cuda)

        tokenizer = AutoTokenizer.from_pretrained(self.name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.name,
            torch_dtype=torch_dtype,
            use_safetensors=True,  # torch<2.6 + transformers CVE-2025-32434 guard blocks .bin/torch.load
        )
        # SINGLE-CARD ASSUMPTION: pin the whole model to cuda:0 rather than
        # device_map="auto". Server runs expose exactly one GPU via Docker
        # --gpus device=N (scripts/docker/run.sh), so cuda:0 IS that card; SAE
        # (models/sae.py) and NLI (models/nli.py) also use .cuda() == cuda:0, so
        # everything stays co-resident and the intervene hook does no cross-device
        # hops. To shard a large model across MULTIPLE cards, revert to
        # device_map="auto" and make SAE/NLI placement device-aware — see TODO.md
        # "Multi-GPU".
        if has_cuda:
            model = model.to("cuda:0")
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
        seed: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        batch_size: int = 0,
    ) -> list[list[Generation]]:
        import torch

        self._ensure_loaded()
        do_sample = temperature > 1e-5
        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": n,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            _add_truncation_kwargs(gen_kwargs, top_p, top_k)

        # Re-seed torch's global RNG ONCE before the (possibly chunked) sampled
        # run so that the same (cfg.seed, prompt, n) yields the same sample set
        # across runs and across before/after intervention sweeps. Seeding once
        # (not per chunk) keeps the RNG stream monotonic; with batch_size=0
        # there is a single chunk so this is byte-identical to the old path.
        # No-op for greedy.
        if seed is not None and do_sample:
            torch.manual_seed(int(seed))

        # Chunk the prompt list so the effective GPU batch (chunk * n) is bounded
        # by `batch_size`, not by len(prompts). batch_size<=0 → one chunk (all
        # prompts), preserving the pre-batching behaviour exactly. Output order
        # is concatenated chunk-by-chunk, so it matches the single-batch path.
        step = len(prompts) if batch_size <= 0 else int(batch_size)
        step = max(1, step)
        result: list[list[Generation]] = []
        for start in range(0, len(prompts), step):
            chunk = prompts[start : start + step]
            result.extend(self._generate_chunk(chunk, n=n, system=system, gen_kwargs=gen_kwargs))
        return result

    def _generate_chunk(
        self, prompts: list[str], *, n: int, system: str | None, gen_kwargs: dict
    ) -> list[list[Generation]]:
        """Tokenise, generate, and decode one chunk of prompts in one forward.

        The `decoded[i * n + j]` index math is chunk-local, so concatenating the
        per-chunk results reproduces the order of a single full-batch call.
        """
        import torch

        texts = [self._apply_chat_template(p, system=system) for p in prompts]
        prev_side = self._tokenizer.padding_side
        self._tokenizer.padding_side = "left"
        try:
            inputs = self._tokenizer(texts, return_tensors="pt", padding=True).to(self._device)
        finally:
            self._tokenizer.padding_side = prev_side

        input_len = inputs.input_ids.shape[1]
        with torch.inference_mode():
            out = self._model.generate(**inputs, **gen_kwargs)

        decoded = self._tokenizer.batch_decode(out[:, input_len:], skip_special_tokens=True)
        return [
            [Generation(text=decoded[i * n + j], finish_reason="stop") for j in range(n)]
            for i in range(len(prompts))
        ]

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

    def hidden_states(
        self, texts: list[str], *, dtype: str = "float32"
    ) -> "Iterator[torch.Tensor]":
        """Forward each text once; yield [n_layers+1, seq_len, d_model] tensors.

        Index 0 is the token-embedding output; indices 1..n_layers are
        transformer-block residual-stream outputs. Tensors are yielded ONE AT A
        TIME on CPU in `dtype` (default float32) so the caller can stream-save
        without holding all N in memory. `dtype` ∈ {float32, bfloat16, float16};
        bfloat16 halves the host-RAM/disk footprint at no downstream cost.
        """
        import torch

        self._ensure_loaded()
        out_dtype = getattr(torch, _DTYPE_MAP[dtype])
        for text in texts:
            inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
            with torch.inference_mode():
                out = self._model(**inputs, output_hidden_states=True, return_dict=True)
            # tuple of (n_layers+1) tensors, each shape [1, seq_len, d_model]
            stacked = torch.stack(out.hidden_states, dim=0).squeeze(1)
            yield stacked.to(out_dtype).cpu()

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
        hook_layer: int | list[int],
        hook_fn,
        temperature: float,
        max_new_tokens: int,
        n: int = 1,
        system: str | None = None,
        seed: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> list[list[Generation]]:
        """Same as `generate` but with `hook_fn` applied at the residual stream of `hook_layer`.

        `hook_layer` accepts either a single int or a list of ints (paper App E.1
        intervenes on a contiguous range). `hook_fn` may be:
          - a single Callable → applied to every layer in `hook_layer`,
          - or a Mapping[int, Callable] → per-layer Callables keyed by index.

        `hook_fn(residual: [B, T, D]) -> [B, T, D]` runs after every forward of
        each target transformer block. Path is Llama/Mistral/Qwen2-compatible
        (`model.model.layers[i]`).
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
            _add_truncation_kwargs(gen_kwargs, top_p, top_k)

        handles = self._register_residual_hooks(hook_layer, hook_fn)
        try:
            if seed is not None and do_sample:
                torch.manual_seed(int(seed))
            with torch.inference_mode():
                out = self._model.generate(**inputs, **gen_kwargs)
        finally:
            for h in handles:
                h.remove()

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

    # ----------------------------------------------------------------- #
    # Shared hook registration                                           #
    # ----------------------------------------------------------------- #

    def _register_residual_hooks(self, hook_layer, hook_fn) -> list:
        """Attach a forward hook to every requested transformer block.

        `hook_layer` accepts int or list[int]; `hook_fn` is either a callable
        applied to every layer or a Mapping[int, Callable] keyed by layer.
        Returns the list of hook handles — caller is responsible for `.remove()`.

        Hardcoded `model.model.layers[i]` path: Llama / Mistral / Qwen2 / Gemma2.
        """
        if hook_layer is None or hook_fn is None:
            return []
        layers = (
            [int(hook_layer)]
            if isinstance(hook_layer, int)
            else [int(l) for l in hook_layer]
        )

        def _layer_fn(layer: int):
            if isinstance(hook_fn, dict):
                return hook_fn[layer]
            return hook_fn

        def _make_wrapper(fn):
            def _wrapped(module, _inputs, output):
                if isinstance(output, tuple):
                    return (fn(output[0]),) + output[1:]
                return fn(output)
            return _wrapped

        return [
            self._model.model.layers[l].register_forward_hook(_make_wrapper(_layer_fn(l)))
            for l in layers
        ]

    # ----------------------------------------------------------------- #
    # Teacher-forced NLL (for perplexity diagnostics)                    #
    # ----------------------------------------------------------------- #

    def forward_nll_with_hook(
        self,
        text: str,
        *,
        hook_layer: int | list[int] | None = None,
        hook_fn=None,
        stride: int = 512,
    ) -> tuple[float, int]:
        """Sliding-window NLL of `text` under teacher forcing, with optional hook.

        Returns `(sum_nll, n_tokens)`. Perplexity is `exp(sum_nll / n_tokens)`.
        Stride overlap is masked with -100 per the HF perplexity recipe so the
        same token isn't counted twice. softmax/log run in float32 even if the
        model weights are bf16/fp16, to avoid tail-probability collapse.

        Hook is applied for every forward (no decode-step skip — there is no
        generation here, every position is "prefill").
        """
        import torch

        self._ensure_loaded()
        encoding = self._tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoding.input_ids.to(self._device)
        n_full = int(input_ids.shape[1])
        max_len = int(getattr(self._model.config, "max_position_embeddings", 2048))
        # Pick a window size that is comfortable for short inputs and bounded
        # by the model context. Stride controls the step between windows.
        window = min(max(stride * 2, 256), max_len, n_full if n_full > 0 else 1)
        stride = max(1, min(stride, window))

        handles = self._register_residual_hooks(hook_layer, hook_fn)
        sum_nll = 0.0
        n_tokens = 0
        try:
            with torch.inference_mode():
                prev_end = 0
                for begin in range(0, n_full, stride):
                    end = min(begin + window, n_full)
                    window_ids = input_ids[:, begin:end]
                    if window_ids.shape[1] < 2:
                        break
                    target_ids = window_ids.clone()
                    # Mask the overlap with the previous window: only score
                    # tokens that haven't been scored yet (HF recipe).
                    trg_len = end - prev_end
                    target_ids[:, :-trg_len] = -100
                    out = self._model(window_ids, return_dict=True)
                    logits = out.logits[:, :-1, :].float()
                    labels = target_ids[:, 1:]
                    mask = labels != -100
                    if not bool(mask.any()):
                        prev_end = end
                        if end == n_full:
                            break
                        continue
                    log_probs = torch.log_softmax(logits, dim=-1)
                    safe_labels = labels.clamp(min=0)
                    nll = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
                    nll = nll * mask
                    sum_nll += float(nll.sum().item())
                    n_tokens += int(mask.sum().item())
                    prev_end = end
                    if end == n_full:
                        break
        finally:
            for h in handles:
                h.remove()
        return sum_nll, n_tokens

    # ----------------------------------------------------------------- #
    # KL divergence between baseline and hooked distributions            #
    # ----------------------------------------------------------------- #

    def forward_kl_with_hook(
        self,
        prompts: list[str],
        *,
        hook_layer: int | list[int],
        hook_fn,
    ) -> dict[str, float]:
        """KL(p_baseline || p_intervened) on `prompts`.

        Runs two forwards per prompt (without and with hook). softmax/log
        computed in float32. Reports:

        * `mean_kl_all_positions` — averaged over every non-pad position.
        * `mean_kl_last_prompt_token` — KL at the position that produces the
          first generated token (the QA-relevant slot).
        * `top1_disagreement_rate` — fraction of positions where argmax shifts.
        * `top5_mass_delta` — mean of `|sum top-5 p_int − sum top-5 p_base|`,
          a cheap signal for whether the distribution collapsed/spread.

        Padding is right-aligned for position-aligned comparison.
        """
        import torch

        self._ensure_loaded()
        prev_side = self._tokenizer.padding_side
        self._tokenizer.padding_side = "right"
        try:
            inputs = self._tokenizer(
                prompts, return_tensors="pt", padding=True, add_special_tokens=True
            ).to(self._device)
        finally:
            self._tokenizer.padding_side = prev_side

        input_ids = inputs.input_ids
        attn = inputs.attention_mask.bool()

        with torch.inference_mode():
            baseline_out = self._model(input_ids, attention_mask=inputs.attention_mask, return_dict=True)
            baseline_logits = baseline_out.logits.float()

            handles = self._register_residual_hooks(hook_layer, hook_fn)
            try:
                hooked_out = self._model(input_ids, attention_mask=inputs.attention_mask, return_dict=True)
                hooked_logits = hooked_out.logits.float()
            finally:
                for h in handles:
                    h.remove()

        log_p = torch.log_softmax(baseline_logits, dim=-1)
        log_q = torch.log_softmax(hooked_logits, dim=-1)
        p = log_p.exp()
        # KL(P || Q) = Σ p · (log p - log q), in nats.
        kl = (p * (log_p - log_q)).sum(dim=-1)  # [B, T]
        kl = torch.clamp(kl, min=0.0)  # numerical floor; KL is non-negative.
        argmax_p = baseline_logits.argmax(dim=-1)
        argmax_q = hooked_logits.argmax(dim=-1)
        disagree = (argmax_p != argmax_q).float()  # [B, T]

        # Top-5 mass shift.
        p_soft = torch.softmax(baseline_logits, dim=-1)
        q_soft = torch.softmax(hooked_logits, dim=-1)
        top_p = p_soft.topk(min(5, p_soft.shape[-1]), dim=-1).values.sum(dim=-1)
        top_q = q_soft.topk(min(5, q_soft.shape[-1]), dim=-1).values.sum(dim=-1)
        top5_delta = (top_q - top_p).abs()

        mask = attn.float()
        n_positions = float(mask.sum().item())
        if n_positions == 0:
            return {
                "mean_kl_all_positions": 0.0,
                "mean_kl_last_prompt_token": 0.0,
                "top1_disagreement_rate": 0.0,
                "top5_mass_delta": 0.0,
                "n_prompts": int(len(prompts)),
                "n_tokens": 0,
            }

        mean_kl_all = float((kl * mask).sum().item() / n_positions)
        disagree_rate = float((disagree * mask).sum().item() / n_positions)
        top5_mean = float((top5_delta * mask).sum().item() / n_positions)

        # Last prompt-token: attn.sum(-1) - 1 is its index per sequence.
        last_idx = attn.long().sum(dim=-1) - 1
        last_idx = last_idx.clamp(min=0)
        batch_idx = torch.arange(kl.shape[0], device=kl.device)
        last_kl = kl[batch_idx, last_idx]
        mean_kl_last = float(last_kl.mean().item())

        return {
            "mean_kl_all_positions": mean_kl_all,
            "mean_kl_last_prompt_token": mean_kl_last,
            "top1_disagreement_rate": disagree_rate,
            "top5_mass_delta": top5_mean,
            "n_prompts": int(len(prompts)),
            "n_tokens": int(n_positions),
        }
