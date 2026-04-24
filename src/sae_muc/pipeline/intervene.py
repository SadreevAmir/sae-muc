"""intervene: forward-hook VUF intervention on the residual stream.

Two modes (selected via `cfg.stages.intervene.mode`):

* **fixed**: for each α in `alpha_grid`, add α·r_VU^(l) at layer l to every
  token, generate answers with the usual greedy+samples protocol. Per-α
  generations land in `intervention/alpha_{a:+.2f}/generations.parquet`.
  This is the paper's Fig.5/6 ablation sweep.

* **adaptive** (Mechanistic Uncertainty Calibration, paper §4.2):
  per-question α_su(x) = clip(SU_norm(x) − VU(x), 0, α_max)
  where SU_norm is min-max normalised semantic entropy over the run and
  VU is the mean judge VU over the N sampled answers. We loop prompts
  one at a time, build a constant-α hook with that question's α, run the
  same greedy+samples pair. Output:
    intervention/adaptive/generations.parquet  — rows carry per-question α
    intervention/adaptive/alphas.parquet       — per-question (vu, se,
                                                 su_norm, alpha)

Summary meta (paths, layer, method, mode) in `intervention/meta.parquet`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from sae_muc.data.prompts import format_answer_prompt
from sae_muc.pipeline.context import PipelineContext

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

OUTPUT_META = "intervention/meta.parquet"
ADAPTIVE_DIR = "intervention/adaptive"


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


def _build_hook_dispatch(method: str, direction: "torch.Tensor", alpha: float, sae):
    if method == "linear_vuf":
        return _build_hook(direction, alpha)
    if method == "sae_projected":
        return _build_sae_projected_hook(direction, sae, alpha)
    if method in ("sae_emd", "sae_clamp"):
        raise NotImplementedError(
            f"intervene.method={method!r} needs the sae_features stage to "
            "select uncertainty/certainty feature indices; this lands in a "
            "follow-up commit."
        )
    raise NotImplementedError(f"Unknown intervene.method={method!r}")


def _compute_adaptive_alphas(
    ctx: PipelineContext,
    sample_ids: list[str],
    alpha_max: float,
) -> pd.DataFrame:
    """Per-question α via Eq. 6: α = clip(SU_norm(x) − VU(x), 0, α_max).

    Returns a DataFrame in `sample_ids` order with columns
    (sample_id, vu, se, su_norm, alpha).
    """
    judge = ctx.store.load_parquet("judge_scores.parquet")
    vu_per_q = judge[judge["kind"] == "sample"].groupby("sample_id")["vu_score"].mean()
    se = ctx.store.load_parquet("semantic_entropy.parquet").set_index("sample_id")[
        "semantic_entropy"
    ]

    su = np.asarray([float(se.loc[sid]) for sid in sample_ids], dtype=float)
    vu = np.asarray([float(vu_per_q.loc[sid]) for sid in sample_ids], dtype=float)

    su_min, su_max = float(su.min()), float(su.max())
    su_range = max(su_max - su_min, 1e-8)
    su_norm = (su - su_min) / su_range  # 0..1

    alpha = np.clip(su_norm - vu, 0.0, float(alpha_max))
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "vu": vu,
            "se": su,
            "su_norm": su_norm,
            "alpha": alpha,
        }
    )


def _run_fixed(
    ctx: PipelineContext,
    prompts: list[str],
    sample_ids: list[str],
    direction: "torch.Tensor",
    target_layer: int,
) -> list[str]:
    cfg = ctx.cfg.stages.intervene
    gen_cfg = ctx.cfg.stages.generate
    outputs: list[str] = []
    for alpha in cfg.alpha_grid:
        hook = _build_hook_dispatch(cfg.method, direction, alpha, ctx.sae)
        greedy = ctx.llm.generate_with_hook(
            prompts, hook_layer=target_layer, hook_fn=hook,
            temperature=gen_cfg.temperature_low, max_new_tokens=gen_cfg.max_new_tokens, n=1,
        )
        sampled = ctx.llm.generate_with_hook(
            prompts, hook_layer=target_layer, hook_fn=hook,
            temperature=gen_cfg.temperature_high, max_new_tokens=gen_cfg.max_new_tokens,
            n=gen_cfg.n_samples,
        )
        rows = _rows_for_generations(sample_ids, greedy, sampled, alpha=float(alpha))
        path = f"{_alpha_dir(alpha)}/generations.parquet"
        ctx.store.save_parquet(path, pd.DataFrame(rows))
        outputs.append(path)

    meta = pd.DataFrame(
        {
            "alpha": [float(a) for a in cfg.alpha_grid],
            "path": [f"{_alpha_dir(a)}/generations.parquet" for a in cfg.alpha_grid],
            "layer": [target_layer] * len(cfg.alpha_grid),
            "method": [cfg.method] * len(cfg.alpha_grid),
            "mode": ["fixed"] * len(cfg.alpha_grid),
        }
    )
    ctx.store.save_parquet(OUTPUT_META, meta)
    outputs.append(OUTPUT_META)
    return outputs


def _run_adaptive(
    ctx: PipelineContext,
    prompts: list[str],
    sample_ids: list[str],
    direction: "torch.Tensor",
    target_layer: int,
) -> list[str]:
    cfg = ctx.cfg.stages.intervene
    gen_cfg = ctx.cfg.stages.generate

    alphas_df = _compute_adaptive_alphas(ctx, sample_ids, cfg.alpha_max)
    ctx.store.save_parquet(f"{ADAPTIVE_DIR}/alphas.parquet", alphas_df)

    # One prompt at a time: build a constant-α hook with that question's α and
    # run the greedy + sampled generate pair. Slower than batching, but keeps
    # the hook code dead-simple and side-steps per-sequence α bookkeeping
    # across num_return_sequences replication. Batched per-sample α is in TODO.
    rows: list[dict] = []
    for i, (sid, prompt) in enumerate(zip(sample_ids, prompts, strict=True)):
        alpha_i = float(alphas_df.iloc[i]["alpha"])
        hook = _build_hook_dispatch(cfg.method, direction, alpha_i, ctx.sae)

        greedy_i = ctx.llm.generate_with_hook(
            [prompt], hook_layer=target_layer, hook_fn=hook,
            temperature=gen_cfg.temperature_low, max_new_tokens=gen_cfg.max_new_tokens, n=1,
        )
        sampled_i = ctx.llm.generate_with_hook(
            [prompt], hook_layer=target_layer, hook_fn=hook,
            temperature=gen_cfg.temperature_high, max_new_tokens=gen_cfg.max_new_tokens,
            n=gen_cfg.n_samples,
        )
        rows.extend(
            _rows_for_generations([sid], greedy_i, sampled_i, alpha=alpha_i)
        )

    gen_path = f"{ADAPTIVE_DIR}/generations.parquet"
    ctx.store.save_parquet(gen_path, pd.DataFrame(rows))

    meta = pd.DataFrame(
        [
            {
                "alpha": None,
                "path": gen_path,
                "layer": target_layer,
                "method": cfg.method,
                "mode": "adaptive",
                "mean_alpha": float(alphas_df["alpha"].mean()),
                "min_alpha": float(alphas_df["alpha"].min()),
                "max_alpha": float(alphas_df["alpha"].max()),
                "alpha_max": float(cfg.alpha_max),
            }
        ]
    )
    ctx.store.save_parquet(OUTPUT_META, meta)
    return [f"{ADAPTIVE_DIR}/alphas.parquet", gen_path, OUTPUT_META]


def _rows_for_generations(sample_ids, greedy, sampled, *, alpha: float) -> list[dict]:
    rows: list[dict] = []
    for sid, g_list, s_list in zip(sample_ids, greedy, sampled, strict=True):
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
    return rows


def run(ctx: PipelineContext) -> list[str]:
    cfg = ctx.cfg.stages.intervene

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
    sample_ids = list(samples["sample_id"])
    prompts = [format_answer_prompt(q, eliciting=True) for q in samples["question"]]

    if cfg.mode == "adaptive":
        log.info(
            "mode=adaptive, method=%s, layer=%d, α_max=%.2f (per-question α via Eq.6)",
            cfg.method, target_layer, cfg.alpha_max,
        )
        return _run_adaptive(ctx, prompts, sample_ids, direction, target_layer)
    log.info(
        "mode=fixed, method=%s, layer=%d, α grid=%s",
        cfg.method, target_layer, list(cfg.alpha_grid),
    )
    return _run_fixed(ctx, prompts, sample_ids, direction, target_layer)
