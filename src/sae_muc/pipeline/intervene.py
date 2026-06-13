"""intervene: forward-hook VUF intervention on the residual stream.

Two modes (selected via `cfg.stages.intervene.mode`):

* **fixed**: for each α in `alpha_grid`, add α·r_VU^(l) at every layer in
  `cfg.stages.intervene.layer` (paper App E.1 typically uses a contiguous
  range like Llama 15-31) to every token, generate answers with the usual
  greedy+samples protocol. Per-α generations land in
  `intervention/alpha_{a:+.2f}/generations.parquet`. This is the paper's
  Fig.5/6 ablation sweep.

* **adaptive** (Mechanistic Uncertainty Calibration, paper §4.2):
  per-question α_su(x) = clip((SU_norm(x) − VU(x))·α_max, 0, α_max)
  where SU_norm = SE / ln(N) (paper App G.1; N is the number of sampled
  answers used to estimate SE) and VU is the mean judge VU over those
  N sampled answers. We loop prompts one at a time, build a constant-α
  hook with that question's α, run the same greedy+samples pair. Output:
    intervention/adaptive/generations.parquet  — rows carry per-question α
    intervention/adaptive/alphas.parquet       — per-question (vu, se,
                                                 su_norm, alpha)

Summary meta (paths, layers, method, mode) in `intervention/meta.parquet`.

Per-layer SAE. `cfg.stages.intervene.layer` may be a list (paper App E.1
ranges, e.g. Llama 15-31). For SAE-based methods (`sae_emd`, `sae_clamp`,
`sae_projected`) the hook at each layer uses the SAE registered for that
specific layer in `ctx.saes` — Gemma-Scope and Llama-Scope are trained
per residual layer, so reusing one SAE across layers produces OOD noise.
Validation against the sae-lens release happens in `run()` via
`assert_sae_layers_available` before any forward.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from sae_muc.data.prompts import format_answer_prompt
from sae_muc.models.sae import assert_sae_layers_available
from sae_muc.pipeline._utils import PROMPT_PLAIN, _resolve_layers, select_prompt_kind
from sae_muc.pipeline.context import PipelineContext
from sae_muc.pipeline.paper_layer_ranges import paper_max_alpha

_SAE_METHODS = ("sae_emd", "sae_clamp", "sae_projected")

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

OUTPUT_META = "intervention/meta.parquet"
ADAPTIVE_DIR = "intervention/adaptive"


def _alpha_dir(alpha: float) -> str:
    return f"intervention/alpha_{alpha:+.2f}"


def _adaptive_amax_dir(alpha_max: float) -> str:
    return f"intervention/adaptive_amax_{alpha_max:+.2f}"


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


def _build_sae_emd_hook(
    uncertainty_idx,
    certainty_idx,
    sae,
    alpha,
    *,
    cohen_d=None,
    delta_mode: str = "cohen_d",
):
    """`sae_emd`: f' = f + α·δ, h' = decode(f') + err.

    δ is shaped per `delta_mode`:
      - "multihot"  → +1 at every uncertainty feature, -1 at every certainty
        feature. Every selected feature is pushed uniformly.
      - "cohen_d"   → δ[i] = cohen_d[i] for both groups (positive for unc,
        negative for cert by construction), then L2-normalised so α has a
        comparable scale across runs (paper / old prototype default).

    α>0 increases uncertainty-feature activations and decreases certainty ones.
    """
    import torch

    delta = torch.zeros(sae.d_latent, dtype=torch.float32)
    if delta_mode == "multihot":
        for idx in uncertainty_idx:
            delta[idx] = 1.0
        for idx in certainty_idx:
            delta[idx] = -1.0
    elif delta_mode == "cohen_d":
        if cohen_d is None:
            raise ValueError("delta_mode='cohen_d' needs the cohen_d mapping per feature.")
        for idx in uncertainty_idx:
            delta[idx] = float(cohen_d[idx])
        for idx in certainty_idx:
            delta[idx] = float(cohen_d[idx])
        norm = float(delta.norm().item())
        if norm > 1e-8:
            delta = delta / norm
    else:
        raise ValueError(f"Unknown sae_emd_delta mode: {delta_mode!r}")

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


def _build_sae_clamp_hook(
    uncertainty_idx,
    certainty_idx,
    sae,
    alpha,
    *,
    unc_targets,
):
    """`sae_clamp`: soft-push uncertainty features toward their per-feature
    target activation; soft-suppress certainty features toward zero.

    Per old prototype hooks._apply_sae_clamp:
      α'      = clip(α, 0, 1)
      f'[unc] = f[unc] + α' · max(0, target[unc] − f[unc])     (push up)
      f'[cer] = f[cer] · (1 − α')                              (suppress)
      h'      = decode(f') + (h − decode(f))

    `unc_targets` is a Mapping[idx → mean_uncertain[idx]] derived from
    `sae_features/stats.parquet` — one absolute target per feature.
    """
    import torch

    a = float(min(max(alpha, 0.0), 1.0))
    target_vec = torch.zeros(sae.d_latent, dtype=torch.float32)
    for idx in uncertainty_idx:
        target_vec[idx] = float(unc_targets[idx])

    def hook_fn(residual: "torch.Tensor") -> "torch.Tensor":
        orig_shape = residual.shape
        orig_dtype = residual.dtype
        flat = residual.reshape(-1, orig_shape[-1]).to(dtype=torch.float32)
        f = sae.encode(flat)
        recon = sae.decode(f)
        err = flat - recon
        f_new = f.clone()
        if uncertainty_idx:
            tv = target_vec.to(f.device, dtype=f.dtype)
            current = f_new[..., uncertainty_idx]
            gap = torch.clamp(tv[uncertainty_idx] - current, min=0.0)
            f_new[..., uncertainty_idx] = current + a * gap
        if certainty_idx:
            f_new[..., certainty_idx] = f_new[..., certainty_idx] * (1.0 - a)
        out = sae.decode(f_new) + err
        return out.to(dtype=orig_dtype).reshape(orig_shape)

    return hook_fn


def _build_hook_dispatch(
    method: str,
    direction: "torch.Tensor",
    alpha: float,
    sae,
    *,
    layer_stats: dict | None = None,
    sae_emd_delta: str = "cohen_d",
):
    if method == "linear_vuf":
        return _build_hook(direction, alpha)
    if method == "sae_projected":
        return _build_sae_projected_hook(direction, sae, alpha)
    if method == "sae_emd":
        if layer_stats is None:
            raise ValueError(
                "sae_emd requires `sae_features/stats.parquet` — run the "
                "`sae_features` stage before `intervene`."
            )
        return _build_sae_emd_hook(
            layer_stats["uncertainty_idx"],
            layer_stats["certainty_idx"],
            sae, alpha,
            cohen_d=layer_stats["cohen_d"],
            delta_mode=sae_emd_delta,
        )
    if method == "sae_clamp":
        if layer_stats is None:
            raise ValueError(
                "sae_clamp requires `sae_features/stats.parquet` — run the "
                "`sae_features` stage before `intervene`."
            )
        return _build_sae_clamp_hook(
            layer_stats["uncertainty_idx"],
            layer_stats["certainty_idx"],
            sae, alpha,
            unc_targets=layer_stats["mean_uncertain"],
        )
    raise NotImplementedError(f"Unknown intervene.method={method!r}")


def _load_sae_feature_stats(
    ctx: PipelineContext, layers: list[int]
) -> dict[int, dict]:
    """For each requested layer, return per-feature SAE stats needed by hooks.

    Returned dict has keys:
      uncertainty_idx, certainty_idx — list[int]
      cohen_d: dict[int, float]      — cohen_d for every selected feature
                                       (used by sae_emd's cohen_d delta).
      mean_uncertain: dict[int, float] — per-feature target activation
                                         (used by sae_clamp's soft-push).
    """
    stats = ctx.store.load_parquet("sae_features/stats.parquet")
    out: dict[int, dict] = {}
    for layer in layers:
        sub = stats[stats["layer"] == layer]
        if sub.empty:
            raise ValueError(
                f"sae_features/stats.parquet has no rows for layer={layer}; "
                f"available layers: {sorted(set(int(x) for x in stats['layer'].tolist()))}"
            )
        unc_rows = sub[sub["selected_as"] == "uncertainty"]
        cer_rows = sub[sub["selected_as"] == "certainty"]
        unc = [int(x) for x in unc_rows["feature_id"].tolist()]
        cer = [int(x) for x in cer_rows["feature_id"].tolist()]
        cohen_d_map = {int(r["feature_id"]): float(r["cohen_d"]) for _, r in sub.iterrows()}
        mean_uncertain_map = {int(r["feature_id"]): float(r["mean_uncertain"]) for _, r in sub.iterrows()}
        out[layer] = {
            "uncertainty_idx": unc,
            "certainty_idx": cer,
            "cohen_d": cohen_d_map,
            "mean_uncertain": mean_uncertain_map,
        }
    return out


def _skip_decode_step(hook_fn):
    """P1 wrapper: pass through residual untouched when seq_len == 1.

    Autoregressive .generate() forwards one token at a time after prefill;
    the residual then has shape [B, 1, D]. Old_prototype's
    apply_during_generation=False mode kept the steering on prefill only,
    matching the original Fig.5/6 ablation set-up.
    """
    def wrapped(residual):
        if residual.shape[1] == 1:
            return residual
        return hook_fn(residual)
    return wrapped


def build_per_layer_hooks(
    *,
    method: str,
    sae_emd_delta: str,
    apply_during_generation: bool,
    layers: list[int],
    directions: dict[int, "torch.Tensor"],
    alpha: float,
    saes: dict[int, "object"],
    layer_stats: dict[int, dict] | None = None,
) -> dict[int, callable]:
    """Build a `{layer: hook_fn}` mapping with explicit args (no ctx).

    Pure-args sibling of `_build_per_layer_hooks`. Stages other than
    `intervene` (e.g. `diagnostics`) call this directly with the variant
    parameters parsed from `intervention/meta.parquet` — calling the
    ctx-wrapper would silently reuse the current config's `method` and
    `sae_emd_delta`, not the variant's.

    `layer_stats` is required for SAE methods (`sae_emd`, `sae_clamp`) and
    ignored otherwise. Each entry has the same shape produced by
    `_load_sae_feature_stats`: keys `uncertainty_idx`, `certainty_idx`,
    `cohen_d`, `mean_uncertain`.
    """
    hooks: dict[int, callable] = {}
    for layer in layers:
        stats = layer_stats.get(layer) if layer_stats is not None else None
        hook_fn = _build_hook_dispatch(
            method, directions[layer], alpha, saes[layer],
            layer_stats=stats,
            sae_emd_delta=sae_emd_delta,
        )
        if not apply_during_generation:
            hook_fn = _skip_decode_step(hook_fn)
        hooks[layer] = hook_fn
    return hooks


def _build_per_layer_hooks(
    ctx: PipelineContext,
    layers: list[int],
    directions: dict[int, "torch.Tensor"],
    alpha: float,
) -> dict[int, callable]:
    """ctx-wrapper around `build_per_layer_hooks`: reads method/delta from cfg."""
    cfg = ctx.cfg.stages.intervene
    needs_features = cfg.method in ("sae_emd", "sae_clamp")
    layer_stats = _load_sae_feature_stats(ctx, layers) if needs_features else None
    return build_per_layer_hooks(
        method=cfg.method,
        sae_emd_delta=cfg.sae_emd_delta,
        apply_during_generation=cfg.apply_during_generation,
        layers=layers,
        directions=directions,
        alpha=alpha,
        saes=ctx.saes,
        layer_stats=layer_stats,
    )


def _log_sae_feature_summary(ctx: PipelineContext, layers: list[int]) -> None:
    """N2: emit a one-line per-layer summary of selected SAE features."""
    cfg = ctx.cfg.stages.intervene
    if cfg.method not in ("sae_emd", "sae_clamp"):
        return
    per_layer = _load_sae_feature_stats(ctx, layers)
    for layer in layers:
        s = per_layer[layer]
        log.info(
            "SAE features: %d uncertainty, %d certainty (layer %d, method=%s)",
            len(s["uncertainty_idx"]), len(s["certainty_idx"]), layer, cfg.method,
        )


def _resolve_alpha_max(alpha_max_cfg: float | str, model_name: str) -> float:
    """Resolve cfg.stages.intervene.alpha_max ("paper" → per-model App G.1)."""
    if alpha_max_cfg == "paper":
        return paper_max_alpha(model_name)
    return float(alpha_max_cfg)


def _resolve_alpha_max_list(
    alpha_max_cfg: float | str | list, model_name: str
) -> list[float]:
    """Resolve cfg.alpha_max to an ORDERED list of positive float ceilings.

    A scalar (float or "paper") becomes a 1-element list. Each list entry is
    resolved via `_resolve_alpha_max`; duplicates are dropped AFTER resolution
    on the per-ceiling DIRECTORY key (not the raw float), preserving
    first-occurrence order. Deduping on the dir keeps two ceilings that round
    to the same `_adaptive_amax_dir` (e.g. 0.201 and 0.204 → adaptive_amax_+0.20)
    from silently overwriting each other's artefacts / emitting duplicate meta
    paths (e.g. [0.4, "paper"] on a Mistral model — both 0.4 — collapses to one).
    """
    raw = alpha_max_cfg if isinstance(alpha_max_cfg, list) else [alpha_max_cfg]
    out: list[float] = []
    seen_dirs: set[str] = set()
    for entry in raw:
        value = _resolve_alpha_max(entry, model_name)
        variant_dir = _adaptive_amax_dir(value)
        if variant_dir not in seen_dirs:
            seen_dirs.add(variant_dir)
            out.append(value)
    return out


def _compute_adaptive_alphas(
    ctx: PipelineContext,
    sample_ids: list[str],
    alpha_max: float,
) -> pd.DataFrame:
    """Per-question α via Eq.6: α = clip((SU_norm(x) − VU(x))·α_max, 0, α_max).

    α_max is a multiplicative GAIN, not a clip ceiling: the gap is scaled by
    α_max first, then clipped to [0, α_max]. Source of truth for this form is
    the official author code (facebookresearch/verbal_uncertainty_feature_
    calibration, calibration/semantic_control.py).

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

    # reindex (not per-sid .loc): a question missing from the SE/judge frames
    # (all its generations or judge calls failed) yields NaN instead of a
    # KeyError. su NaN -> su_norm 0 below; vu NaN -> handled after the clip.
    su = se.reindex(sample_ids).to_numpy(dtype=float)
    vu = vu_per_q.reindex(sample_ids).to_numpy(dtype=float)
    n_samples = n_samples_col.reindex(sample_ids).to_numpy(dtype=float)

    log_n = np.where(n_samples > 1, np.log(np.maximum(n_samples, 2.0)), 0.0)
    su_norm = np.where(log_n > 0, su / log_n, 0.0)

    alpha = np.clip((su_norm - vu) * float(alpha_max), 0.0, float(alpha_max))
    # A question whose VU/SE could not be measured (judge entirely failed, or it
    # is absent from the SE frame) yields NaN alpha. Steer it with alpha=0 (no
    # intervention, reuse baseline) instead of adding NaN*direction to the
    # residual stream, which would NaN the whole forward and emit garbage.
    n_alpha_nan = int(np.isnan(alpha).sum())
    if n_alpha_nan:
        log.warning(
            "adaptive alpha: %d/%d questions had unmeasurable VU/SE -> alpha=0 (no steering)",
            n_alpha_nan, len(alpha),
        )
    alpha = np.nan_to_num(alpha, nan=0.0)
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "vu": vu,
            "se": su,
            "su_norm": su_norm,
            "alpha": alpha,
        }
    )


def _layers_str(layers: list[int]) -> str:
    """Compact human-readable layer-list rendering for meta rows / logs."""
    if len(layers) == 1:
        return str(layers[0])
    if layers == list(range(layers[0], layers[-1] + 1)):
        return f"{layers[0]}-{layers[-1]}"
    return ",".join(str(l) for l in layers)


def _run_fixed(
    ctx: PipelineContext,
    prompts: list[str],
    sample_ids: list[str],
    directions: dict[int, "torch.Tensor"],
    target_layers: list[int],
) -> list[str]:
    cfg = ctx.cfg.stages.intervene
    gen_cfg = ctx.cfg.stages.generate
    outputs: list[str] = []
    for alpha in cfg.alpha_grid:
        hooks = _build_per_layer_hooks(ctx, target_layers, directions, alpha)
        greedy = ctx.llm.generate_with_hook(
            prompts, hook_layer=target_layers, hook_fn=hooks,
            temperature=gen_cfg.temperature_low, max_new_tokens=gen_cfg.max_new_tokens, n=1,
            seed=ctx.cfg.seed, top_p=gen_cfg.top_p, top_k=gen_cfg.top_k,
        )
        sampled = ctx.llm.generate_with_hook(
            prompts, hook_layer=target_layers, hook_fn=hooks,
            temperature=gen_cfg.temperature_high, max_new_tokens=gen_cfg.max_new_tokens,
            n=gen_cfg.n_samples, seed=ctx.cfg.seed, top_p=gen_cfg.top_p, top_k=gen_cfg.top_k,
        )
        rows = _rows_for_generations(sample_ids, greedy, sampled, alpha=float(alpha))
        path = f"{_alpha_dir(alpha)}/generations.parquet"
        ctx.store.save_parquet(path, pd.DataFrame(rows))
        outputs.append(path)

    layers_label = _layers_str(target_layers)
    meta = pd.DataFrame(
        {
            "alpha": [float(a) for a in cfg.alpha_grid],
            "path": [f"{_alpha_dir(a)}/generations.parquet" for a in cfg.alpha_grid],
            "layers": [layers_label] * len(cfg.alpha_grid),
            "method": [cfg.method] * len(cfg.alpha_grid),
            "mode": ["fixed"] * len(cfg.alpha_grid),
        }
    )
    ctx.store.save_parquet(OUTPUT_META, meta)
    outputs.append(OUTPUT_META)
    return outputs


def _baseline_rows_for_sid(baseline: pd.DataFrame, sid: str) -> list[dict]:
    """Replicate the baseline generations for `sid` with alpha=0.

    Used by the adaptive run when gate_by_detector flags `sid` as safe: the
    intervention is skipped and the existing baseline rows are reused, which
    matches old_prototype's _skip_generation behaviour.
    """
    sub = baseline[baseline["sample_id"] == sid]
    out: list[dict] = []
    for _, r in sub.iterrows():
        out.append(
            {
                "sample_id": sid,
                "alpha": 0.0,
                "kind": str(r["kind"]),
                "gen_idx": int(r["gen_idx"]),
                "text": r["text"],
                "finish_reason": r.get("finish_reason"),
            }
        )
    return out


def _load_at_risk_set(ctx: PipelineContext) -> set[str]:
    """Read detection.parquet, return {sample_id : is_at_risk}.

    Falls back to "everyone is at risk" with a warning if detection.parquet
    is missing the column (e.g. detect stage skipped on insufficient data).
    """
    detection = ctx.store.load_parquet("detection.parquet")
    if "is_at_risk" not in detection.columns:
        log.warning(
            "intervene: gate_by_detector=True but detection.parquet has no "
            "is_at_risk column — treating every sample as at-risk."
        )
        return set(detection["sample_id"].astype(str))
    return set(detection.loc[detection["is_at_risk"].astype(bool), "sample_id"].astype(str))


def _adaptive_steered_rows(
    ctx: PipelineContext,
    prompts: list[str],
    sample_ids: list[str],
    directions: dict[int, "torch.Tensor"],
    target_layers: list[int],
    alphas_df: pd.DataFrame,
    *,
    at_risk: set[str] | None,
    baseline: pd.DataFrame | None,
) -> list[dict]:
    """Per-sample steered greedy+sampled generation for one ceiling's α frame.

    One prompt at a time: build a constant-α hook with that question's α and
    run the greedy + sampled generate pair. Slower than batching, but keeps
    the hook code dead-simple and side-steps per-sequence α bookkeeping across
    num_return_sequences replication. Batched per-sample α is in TODO.
    """
    gen_cfg = ctx.cfg.stages.generate
    rows: list[dict] = []
    for i, (sid, prompt) in enumerate(zip(sample_ids, prompts, strict=True)):
        if at_risk is not None and sid not in at_risk:
            rows.extend(_baseline_rows_for_sid(baseline, sid))
            continue
        alpha_i = float(alphas_df.iloc[i]["alpha"])
        hooks = _build_per_layer_hooks(ctx, target_layers, directions, alpha_i)

        greedy_i = ctx.llm.generate_with_hook(
            [prompt], hook_layer=target_layers, hook_fn=hooks,
            temperature=gen_cfg.temperature_low, max_new_tokens=gen_cfg.max_new_tokens, n=1,
            seed=ctx.cfg.seed, top_p=gen_cfg.top_p, top_k=gen_cfg.top_k,
        )
        sampled_i = ctx.llm.generate_with_hook(
            [prompt], hook_layer=target_layers, hook_fn=hooks,
            temperature=gen_cfg.temperature_high, max_new_tokens=gen_cfg.max_new_tokens,
            n=gen_cfg.n_samples, seed=ctx.cfg.seed, top_p=gen_cfg.top_p, top_k=gen_cfg.top_k,
        )
        rows.extend(
            _rows_for_generations([sid], greedy_i, sampled_i, alpha=alpha_i)
        )
    return rows


def _run_adaptive(
    ctx: PipelineContext,
    prompts: list[str],
    sample_ids: list[str],
    directions: dict[int, "torch.Tensor"],
    target_layers: list[int],
) -> list[str]:
    cfg = ctx.cfg.stages.intervene

    # Ceiling-independent gate: compute the at-risk set + plain baseline ONCE.
    if cfg.gate_by_detector:
        at_risk = _load_at_risk_set(ctx)
        # Reuse the plain baseline for gated-out (safe) samples — the steered
        # generation is plain, so copy the matching plain rows (App C / §2.15).
        baseline = select_prompt_kind(
            ctx.store.load_parquet("generations.parquet"), PROMPT_PLAIN
        )
    else:
        at_risk = None
        baseline = None

    ceilings = _resolve_alpha_max_list(cfg.alpha_max, ctx.cfg.model.name)
    # Back-compat: a scalar alpha_max keeps the legacy single intervention/adaptive
    # dir + single meta row (byte-identical). A list — even 1-element — uses the
    # per-ceiling intervention/adaptive_amax_{v:+.2f} dirs.
    is_sweep = isinstance(cfg.alpha_max, list)

    layers_label = _layers_str(target_layers)
    outputs: list[str] = []
    meta_rows: list[dict] = []
    for alpha_max in ceilings:
        variant_dir = _adaptive_amax_dir(alpha_max) if is_sweep else ADAPTIVE_DIR

        alphas_df = _compute_adaptive_alphas(ctx, sample_ids, alpha_max)
        if cfg.gate_by_detector:
            # Force α=0 on the alphas frame for samples the detector flagged safe,
            # so downstream consumers (sweeps, plots) see a faithful α distribution.
            gate_mask = alphas_df["sample_id"].isin(at_risk).to_numpy()
            alphas_df.loc[~gate_mask, "alpha"] = 0.0
            log.info(
                "gate_by_detector=True (α_max=%.2f): %d/%d samples at-risk "
                "(skipping intervention on the rest, baseline reused)",
                alpha_max, int(gate_mask.sum()), len(alphas_df),
            )

        ctx.store.save_parquet(f"{variant_dir}/alphas.parquet", alphas_df)

        rows = _adaptive_steered_rows(
            ctx, prompts, sample_ids, directions, target_layers, alphas_df,
            at_risk=at_risk, baseline=baseline,
        )
        gen_path = f"{variant_dir}/generations.parquet"
        ctx.store.save_parquet(gen_path, pd.DataFrame(rows))

        meta_rows.append(
            {
                "alpha": None,
                "path": gen_path,
                "layers": layers_label,
                "method": cfg.method,
                "mode": "adaptive",
                "mean_alpha": float(alphas_df["alpha"].mean()),
                "min_alpha": float(alphas_df["alpha"].min()),
                "max_alpha": float(alphas_df["alpha"].max()),
                "alpha_max": float(alpha_max),
            }
        )
        outputs.extend([f"{variant_dir}/alphas.parquet", gen_path])

    ctx.store.save_parquet(OUTPUT_META, pd.DataFrame(meta_rows))
    outputs.append(OUTPUT_META)
    return outputs


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
    target_layers = _resolve_layers(cfg.layer, available, model_name=ctx.cfg.model.name)

    if cfg.method in _SAE_METHODS:
        assert_sae_layers_available(ctx.cfg.sae, target_layers)

    directions: dict[int, "torch.Tensor"] = {
        layer: ctx.store.load_safetensors(
            f"vuf/direction_layer_{layer}.safetensors"
        )["direction"]
        for layer in target_layers
    }

    samples = ctx.store.load_parquet("samples.parquet")
    sample_ids = list(samples["sample_id"])
    # Steered MUC generation uses the NEUTRAL prompt: §4.2 states MUC needs no
    # "additional system prompt design" — uncertainty is injected mechanically
    # via the VUF, so re-using the hedging/eliciting prompt would double-count
    # (told-to-hedge AND VUF-pushed) and confound the causal attribution (§2.15).
    prompts = [format_answer_prompt(q, eliciting=False) for q in samples["question"]]

    if cfg.gate_by_detector and cfg.mode != "adaptive":
        log.warning(
            "intervene.gate_by_detector=True is only honoured in mode='adaptive' "
            "(fixed-mode α-sweeps are meant to perturb every question uniformly); "
            "ignoring the gate for this run."
        )

    layers_label = _layers_str(target_layers)
    _log_sae_feature_summary(ctx, target_layers)
    if cfg.mode == "adaptive":
        log.info(
            "mode=adaptive, method=%s, layers=%s, α_max=%s (per-question α via Eq.6)",
            cfg.method, layers_label,
            _resolve_alpha_max_list(cfg.alpha_max, ctx.cfg.model.name),
        )
        return _run_adaptive(ctx, prompts, sample_ids, directions, target_layers)
    log.info(
        "mode=fixed, method=%s, layers=%s, α grid=%s",
        cfg.method, layers_label, list(cfg.alpha_grid),
    )
    return _run_fixed(ctx, prompts, sample_ids, directions, target_layers)
