"""intervene: forward-hook VUF intervention on the residual stream.

Fixed-α sweep. For each α in `cfg.stages.intervene.alpha_grid`, add
    α * r_VU^(l)    at layer l = cfg.stages.intervene.layer
to the residual stream of every token, then generate answers with the
same shape the `generate` stage produces (one greedy at T=low plus N
samples at T=high).

Per-α generations land in
    intervention/alpha_{a:+.2f}/generations.parquet
Summary meta (alphas, paths, layer, method) in intervention/meta.parquet.

Adaptive α(x) mode (Eq. 5–6) and SAE-based interventions land in
separate follow-up commits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from sae_muc.data.prompts import format_answer_prompt
from sae_muc.pipeline.context import PipelineContext

if TYPE_CHECKING:
    import torch

OUTPUT_META = "intervention/meta.parquet"


def _alpha_dir(alpha: float) -> str:
    return f"intervention/alpha_{alpha:+.2f}"


def _resolve_layer(layer_cfg: int | str, available: list[int]) -> int:
    if layer_cfg == "auto":
        if not available:
            raise ValueError("intervene.layer='auto' but no VUF directions are available.")
        return available[len(available) // 2]
    layer = int(layer_cfg)
    if layer not in available:
        raise ValueError(
            f"intervene.layer={layer} has no VUF direction; available layers: {available}"
        )
    return layer


def _build_hook(direction: "torch.Tensor", alpha: float):
    """Returns a forward-hook body that adds α·direction to the residual stream."""

    def hook_fn(residual: "torch.Tensor") -> "torch.Tensor":
        return residual + alpha * direction.to(residual.device, dtype=residual.dtype)

    return hook_fn


def _build_sae_projected_hook(direction: "torch.Tensor", sae, alpha: float):
    """`sae_projected`: project VUF into SAE latent space, add α·latent_vuf, decode.

    For each residual-stream activation `h`:
        f    = sae.encode(h)
        err  = h - sae.decode(f)
        h'   = sae.decode(f + α · latent_vuf) + err
    where `latent_vuf = sae.encode(direction)` is precomputed once.
    """
    import torch

    # [1, d_in] → [1, d_latent] → [d_latent]
    latent_vuf = sae.encode(direction.to(dtype=torch.float32).unsqueeze(0))[0]

    def hook_fn(residual: "torch.Tensor") -> "torch.Tensor":
        orig_shape = residual.shape
        orig_dtype = residual.dtype
        flat = residual.reshape(-1, orig_shape[-1]).to(dtype=torch.float32)
        f = sae.encode(flat)
        recon = sae.decode(f)
        err = flat - recon
        f_new = f + alpha * latent_vuf.to(f.device, dtype=f.dtype)
        recon_new = sae.decode(f_new)
        out = recon_new + err
        return out.to(dtype=orig_dtype).reshape(orig_shape)

    return hook_fn


def _build_hook_dispatch(method: str, direction: "torch.Tensor", alpha: float):
    if method == "linear_vuf":
        return _build_hook(direction, alpha)
    if method == "sae_projected":
        from sae_muc.models.sae import build_sae_backend

        d_in = int(direction.shape[-1])
        sae = build_sae_backend("fake", d_in=d_in)
        return _build_sae_projected_hook(direction, sae, alpha)
    if method in ("sae_emd", "sae_clamp"):
        raise NotImplementedError(
            f"intervene.method={method!r} needs a per-feature selection step "
            "(see archive/old-prototype/sae_muc/build_intervention_config_v2.py); "
            "this lands in a follow-up commit."
        )
    raise NotImplementedError(f"Unknown intervene.method={method!r}")


def run(ctx: PipelineContext) -> list[str]:
    cfg = ctx.cfg.stages.intervene
    gen_cfg = ctx.cfg.stages.generate

    # Method validity is enforced inside _build_hook_dispatch; this catches
    # the unimplemented SAE variants early with a clear message.
    if cfg.method in ("sae_emd", "sae_clamp"):
        raise NotImplementedError(
            f"intervene.method={cfg.method!r} is not implemented yet. "
            "Use 'linear_vuf' or 'sae_projected' for now."
        )

    vuf_meta = ctx.store.load_parquet("vuf/meta.parquet")
    available = sorted(int(x) for x in vuf_meta["layer"].tolist())
    target_layer = _resolve_layer(cfg.layer, available)

    direction = ctx.store.load_safetensors(
        f"vuf/direction_layer_{target_layer}.safetensors"
    )["direction"]

    samples = ctx.store.load_parquet("samples.parquet")
    prompts = [format_answer_prompt(q, eliciting=True) for q in samples["question"]]

    outputs: list[str] = []
    for alpha in cfg.alpha_grid:
        hook = _build_hook_dispatch(cfg.method, direction, alpha)

        greedy = ctx.llm.generate_with_hook(
            prompts,
            hook_layer=target_layer,
            hook_fn=hook,
            temperature=gen_cfg.temperature_low,
            max_new_tokens=gen_cfg.max_new_tokens,
            n=1,
        )
        sampled = ctx.llm.generate_with_hook(
            prompts,
            hook_layer=target_layer,
            hook_fn=hook,
            temperature=gen_cfg.temperature_high,
            max_new_tokens=gen_cfg.max_new_tokens,
            n=gen_cfg.n_samples,
        )

        rows: list[dict] = []
        for sid, g_list, s_list in zip(
            samples["sample_id"], greedy, sampled, strict=True
        ):
            rows.append(
                {
                    "sample_id": sid,
                    "alpha": float(alpha),
                    "kind": "greedy",
                    "gen_idx": 0,
                    "text": g_list[0].text,
                    "finish_reason": g_list[0].finish_reason,
                }
            )
            for j, gen in enumerate(s_list):
                rows.append(
                    {
                        "sample_id": sid,
                        "alpha": float(alpha),
                        "kind": "sample",
                        "gen_idx": j,
                        "text": gen.text,
                        "finish_reason": gen.finish_reason,
                    }
                )
        path = f"{_alpha_dir(alpha)}/generations.parquet"
        ctx.store.save_parquet(path, pd.DataFrame(rows))
        outputs.append(path)

    meta = pd.DataFrame(
        {
            "alpha": [float(a) for a in cfg.alpha_grid],
            "path": [f"{_alpha_dir(a)}/generations.parquet" for a in cfg.alpha_grid],
            "layer": [target_layer] * len(cfg.alpha_grid),
            "method": [cfg.method] * len(cfg.alpha_grid),
        }
    )
    ctx.store.save_parquet(OUTPUT_META, meta)
    outputs.append(OUTPUT_META)
    return outputs
