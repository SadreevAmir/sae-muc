"""intervene: forward-hook VUF intervention on the residual stream.

Two modes (selected via `cfg.stages.intervene.mode`):

* **fixed**: for each α in `alpha_grid`, add α·r_VU^(l) at layer l to every
  token, generate answers with the usual greedy+samples protocol. Per-α
  generations land in `intervention/alpha_{a:+.2f}/generations.parquet`.
  This is the paper's Fig.5/6 ablation sweep.

* **adaptive** (Mechanistic Uncertainty Calibration, paper §4.2):
  per-question α_su(x) = clip(SU_norm(x) − VU(x), 0, α_max)
  where SU_norm = SE / ln(N) (paper App G.1; N is the number of sampled
  answers used to estimate SE) and VU is the mean judge VU over those
  N sampled answers. We loop prompts one at a time, build a constant-α
  hook with that question's α, run the same greedy+samples pair. Output:
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
from sae_muc.pipeline._utils import _resolve_layer
from sae_muc.pipeline.context import PipelineContext

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

OUTPUT_META = "intervention/meta.parquet"
ADAPTIVE_DIR = "intervention/adaptive"


def _alpha_dir(alpha: float) -> str:
    return f"intervention/alpha_{alpha:+.2f}"


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


def _build_sae_emd_hook(uncertainty_idx, certainty_idx, sae, alpha):
    """`sae_emd`: f' = f + α·δ, h' = decode(f') + err.

    δ is a multi-hot vector: +1 at each uncertainty feature, -1 at each
    certainty feature. α scales the whole shift, so α>0 increases
    uncertainty-feature activations and decreases certainty ones.
    """
    import torch

    delta = torch.zeros(sae.d_latent, dtype=torch.float32)
    for idx in uncertainty_idx:
        delta[idx] = 1.0
    for idx in certainty_idx:
        delta[idx] = -1.0

    def hook_fn(residual: "torch.Tensor") -> "torch.Tensor":
        orig_shape = residual.shape
        orig_dtype = residual.dtype
        flat = residual.reshape(-1, orig_shape[-1]).to(dtype=torch.float32)
        f = sae.encode(flat)
        recon = sae.decode(f)
        err = flat - recon
        f_new = f + alpha * delta.to(f.device, dtype=f.dtype)
        out = sae.decode(f_new) + err
        return out.to(dtype=orig_dtype).reshape(orig_shape)

    return hook_fn


def _build_sae_clamp_hook(uncertainty_idx, certainty_idx, sae, alpha, target):
    """`sae_clamp`: set uncertainty features to α·target, certainty features to 0.

    Unlike sae_emd, clamp overwrites rather than adds: the selected
    uncertainty features are forced high, the certainty ones are
    suppressed. α modulates the target level; target is an absolute
    activation scale from `cfg.stages.intervene.sae_clamp_target`.
    """
    import torch

    def hook_fn(residual: "torch.Tensor") -> "torch.Tensor":
        orig_shape = residual.shape
        orig_dtype = residual.dtype
        flat = residual.reshape(-1, orig_shape[-1]).to(dtype=torch.float32)
        f = sae.encode(flat)
        recon = sae.decode(f)
        err = flat - recon
        f_new = f.clone()
        scaled_target = alpha * float(target)
        for idx in uncertainty_idx:
            f_new[..., idx] = scaled_target
        for idx in certainty_idx:
            f_new[..., idx] = 0.0
        out = sae.decode(f_new) + err
        return out.to(dtype=orig_dtype).reshape(orig_shape)

    return hook_fn


def _build_hook_dispatch(
    method: str,
    direction: "torch.Tensor",
    alpha: float,
    sae,
    *,
    uncertainty_idx=None,
    certainty_idx=None,
    clamp_target: float = 10.0,
):
    if method == "linear_vuf":
        return _build_hook(direction, alpha)
    if method == "sae_projected":
        return _build_sae_projected_hook(direction, sae, alpha)
    if method == "sae_emd":
        if uncertainty_idx is None or certainty_idx is None:
            raise ValueError(
                "sae_emd requires `sae_features/stats.parquet` — run the "
                "`sae_features` stage before `intervene`."
            )
        return _build_sae_emd_hook(uncertainty_idx, certainty_idx, sae, alpha)
    if method == "sae_clamp":
        if uncertainty_idx is None or certainty_idx is None:
            raise ValueError(
                "sae_clamp requires `sae_features/stats.parquet` — run the "
                "`sae_features` stage before `intervene`."
            )
        return _build_sae_clamp_hook(
            uncertainty_idx, certainty_idx, sae, alpha, target=clamp_target,
        )
    raise NotImplementedError(f"Unknown intervene.method={method!r}")


def _load_sae_feature_indices(ctx: PipelineContext) -> tuple[list[int], list[int]]:
    stats = ctx.store.load_parquet("sae_features/stats.parquet")
    unc = stats.loc[stats["selected_as"] == "uncertainty", "feature_id"].tolist()
    cer = stats.loc[stats["selected_as"] == "certainty", "feature_id"].tolist()
    return [int(x) for x in unc], [int(x) for x in cer]


def _compute_adaptive_alphas(
    ctx: PipelineContext,
    sample_ids: list[str],
    alpha_max: float,
) -> pd.DataFrame:
    """Per-question α via Eq.6: α = clip(SU_norm(x) − VU(x), 0, α_max).

    SU_norm = SE / ln(N) per paper App G.1 (N = number of sampled answers
    used to estimate SE). N is read per-row from `semantic_entropy.parquet`
    so questions with a degenerate sample count don't poison the rest of
    the run; ln(N≤1) is treated as 0 (su_norm=0).

    Returns a DataFrame in `sample_ids` order with columns
    (sample_id, vu, se, su_norm, alpha).
    """
    judge = ctx.store.load_parquet("judge_scores.parquet")
    vu_per_q = judge[judge["kind"] == "sample"].groupby("sample_id")["vu_score"].mean()
    se_df = ctx.store.load_parquet("semantic_entropy.parquet").set_index("sample_id")
    se = se_df["semantic_entropy"]
    n_samples_col = se_df["n_samples"]

    su = np.asarray([float(se.loc[sid]) for sid in sample_ids], dtype=float)
    vu = np.asarray([float(vu_per_q.loc[sid]) for sid in sample_ids], dtype=float)
    n_samples = np.asarray([int(n_samples_col.loc[sid]) for sid in sample_ids], dtype=int)

    log_n = np.where(n_samples > 1, np.log(np.maximum(n_samples, 2)), 0.0)
    su_norm = np.where(log_n > 0, su / log_n, 0.0)

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
    unc_idx, cer_idx = (
        _load_sae_feature_indices(ctx)
        if cfg.method in ("sae_emd", "sae_clamp")
        else (None, None)
    )
    outputs: list[str] = []
    for alpha in cfg.alpha_grid:
        hook = _build_hook_dispatch(
            cfg.method, direction, alpha, ctx.sae,
            uncertainty_idx=unc_idx, certainty_idx=cer_idx,
            clamp_target=cfg.sae_clamp_target,
        )
        greedy = ctx.llm.generate_with_hook(
            prompts, hook_layer=target_layer, hook_fn=hook,
            temperature=gen_cfg.temperature_low, max_new_tokens=gen_cfg.max_new_tokens, n=1,
            seed=ctx.cfg.seed,
        )
        sampled = ctx.llm.generate_with_hook(
            prompts, hook_layer=target_layer, hook_fn=hook,
            temperature=gen_cfg.temperature_high, max_new_tokens=gen_cfg.max_new_tokens,
            n=gen_cfg.n_samples, seed=ctx.cfg.seed,
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
    unc_idx, cer_idx = (
        _load_sae_feature_indices(ctx)
        if cfg.method in ("sae_emd", "sae_clamp")
        else (None, None)
    )

    alphas_df = _compute_adaptive_alphas(ctx, sample_ids, cfg.alpha_max)
    ctx.store.save_parquet(f"{ADAPTIVE_DIR}/alphas.parquet", alphas_df)

    # One prompt at a time: build a constant-α hook with that question's α and
    # run the greedy + sampled generate pair. Slower than batching, but keeps
    # the hook code dead-simple and side-steps per-sequence α bookkeeping
    # across num_return_sequences replication. Batched per-sample α is in TODO.
    rows: list[dict] = []
    for i, (sid, prompt) in enumerate(zip(sample_ids, prompts, strict=True)):
        alpha_i = float(alphas_df.iloc[i]["alpha"])
        hook = _build_hook_dispatch(
            cfg.method, direction, alpha_i, ctx.sae,
            uncertainty_idx=unc_idx, certainty_idx=cer_idx,
            clamp_target=cfg.sae_clamp_target,
        )

        greedy_i = ctx.llm.generate_with_hook(
            [prompt], hook_layer=target_layer, hook_fn=hook,
            temperature=gen_cfg.temperature_low, max_new_tokens=gen_cfg.max_new_tokens, n=1,
            seed=ctx.cfg.seed,
        )
        sampled_i = ctx.llm.generate_with_hook(
            [prompt], hook_layer=target_layer, hook_fn=hook,
            temperature=gen_cfg.temperature_high, max_new_tokens=gen_cfg.max_new_tokens,
            n=gen_cfg.n_samples, seed=ctx.cfg.seed,
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
