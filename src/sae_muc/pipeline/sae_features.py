"""sae_features: SAE-latent feature analysis for the uncertain / certain split.

Encode pooled hidden states (at every intervene layer) through the SAE,
then rank latent features by Cohen's d between the uncertain and certain
question sets. Selection:

  * `topk` (default): the |d|-largest k_top features in each direction
    become "uncertainty" / "certainty" candidates. Deterministic, cheap.
  * `consensus`: the old prototype's robust filter
    (archive/old-prototype/sae_muc/build_intervention_config_v2.py:207-226).
    Bootstrap-stable AND (BH-FDR significant under Welch's t-test OR
    |Cohen's d| > threshold). Slower but more stable for small n.

Downstream intervene methods `sae_emd` and `sae_clamp` read the resulting
selection from `sae_features/stats.parquet`, keyed by `layer`.

Required upstream stages: vuf (for `vuf/splits.parquet`,
`vuf/meta.parquet`) and hidden_states (for the per-sample layer tensor).
SAE comes from `ctx.sae`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from sae_muc.pipeline._utils import _pool, _resolve_layers
from sae_muc.pipeline.context import PipelineContext

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

OUTPUT = "sae_features/stats.parquet"

# The SAE feature analysis is only required by interventions that consume
# the (uncertainty / certainty) feature-index lists. For linear_vuf and
# sae_projected we skip the stage entirely — running it would do two
# useless things: burn an SAE forward pass and potentially blow up on a
# dim mismatch between the (default) FakeSAE and a real model's d_model.
_SAE_METHODS_REQUIRING_FEATURES = ("sae_emd", "sae_clamp")


def _benjamini_hochberg_mask(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """BH step-up FDR mask: True iff the feature passes at level `alpha`.

    Mirrors the helper in old prototype build_intervention_config_v2.py.
    """
    n = pvals.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresholds = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresholds
    if not np.any(passed):
        return np.zeros(n, dtype=bool)
    kmax = int(np.max(np.where(passed)[0]))
    cutoff = ranked[kmax]
    return pvals <= cutoff


def _bootstrap_topk_stability(
    Xu: np.ndarray, Xc: np.ndarray, *, top_k: int, n_boot: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature hit rates across `n_boot` bootstrap resamples.

    Returns (stab_u, stab_c) of shape [d]: the fraction of bootstraps in
    which feature i landed in the top_k of (mean_u - mean_c) (resp. negated).
    """
    d = Xu.shape[1]
    n_u, n_c = Xu.shape[0], Xc.shape[0]
    k = min(top_k, d)
    hit_u = np.zeros(d, dtype=np.int32)
    hit_c = np.zeros(d, dtype=np.int32)
    for _ in range(n_boot):
        bu = Xu[rng.integers(0, n_u, size=n_u)]
        bc = Xc[rng.integers(0, n_c, size=n_c)]
        diff = bu.mean(axis=0) - bc.mean(axis=0)
        hit_u[np.argsort(diff)[-k:]] += 1
        hit_c[np.argsort(-diff)[-k:]] += 1
    return hit_u / float(n_boot), hit_c / float(n_boot)


def _consensus_select(
    f_u_np: np.ndarray, f_c_np: np.ndarray, d_vals: np.ndarray, *,
    sae_feat_cfg, seed: int, layer: int,
) -> tuple[set[int], set[int]]:
    """Old prototype consensus filter: bootstrap-stable AND (FDR-sig OR |d|>thresh)."""
    from scipy.stats import ttest_ind

    rng = np.random.default_rng(seed + int(layer))
    tt = ttest_ind(f_u_np, f_c_np, axis=0, equal_var=False, nan_policy="omit")
    pvals = np.nan_to_num(np.asarray(tt.pvalue, dtype=np.float64), nan=1.0)
    tvals = np.nan_to_num(np.asarray(tt.statistic, dtype=np.float64), nan=0.0)
    sig = _benjamini_hochberg_mask(pvals, alpha=float(sae_feat_cfg.fdr_alpha))

    stab_u, stab_c = _bootstrap_topk_stability(
        f_u_np, f_c_np,
        top_k=int(sae_feat_cfg.k_top),
        n_boot=int(sae_feat_cfg.bootstrap_n),
        rng=rng,
    )

    d_thresh = float(sae_feat_cfg.cohens_d_threshold)
    boot_thresh = float(sae_feat_cfg.bootstrap_freq_threshold)

    unc_consensus = (stab_u >= boot_thresh) & ((sig & (tvals > 0)) | (d_vals > d_thresh))
    cer_consensus = (stab_c >= boot_thresh) & ((sig & (tvals < 0)) | (d_vals < -d_thresh))
    return set(np.where(unc_consensus)[0].tolist()), set(np.where(cer_consensus)[0].tolist())


def _cohens_d(f_u: "torch.Tensor", f_c: "torch.Tensor") -> "torch.Tensor":
    """Per-feature Cohen's d between two groups. Pooled std with unbiased var.

    f_u: [n_u, d_latent]; f_c: [n_c, d_latent] -> [d_latent].
    """
    import torch

    n_u, n_c = f_u.shape[0], f_c.shape[0]
    mean_u = f_u.mean(dim=0)
    mean_c = f_c.mean(dim=0)
    # Sample variance with Bessel's correction; guard against n<2.
    var_u = f_u.var(dim=0, unbiased=True) if n_u > 1 else torch.zeros_like(mean_u)
    var_c = f_c.var(dim=0, unbiased=True) if n_c > 1 else torch.zeros_like(mean_c)
    denom = n_u + n_c - 2
    pooled_var = ((n_u - 1) * var_u + (n_c - 1) * var_c) / max(denom, 1)
    pooled_std = torch.clamp(pooled_var.sqrt(), min=1e-8)
    return (mean_u - mean_c) / pooled_std


def run(ctx: PipelineContext) -> list[str]:
    import torch

    intervene_cfg = ctx.cfg.stages.intervene
    vuf_cfg = ctx.cfg.stages.vuf
    sae_feat_cfg = ctx.cfg.stages.sae_features

    if intervene_cfg.method not in _SAE_METHODS_REQUIRING_FEATURES:
        log.info(
            "sae_features: skipped (intervene.method=%s does not consume SAE feature lists)",
            intervene_cfg.method,
        )
        return []

    splits = ctx.store.load_parquet("vuf/splits.parquet")
    uncertain_ids = splits[splits["split"] == "uncertain"]["sample_id"].tolist()
    certain_ids = splits[splits["split"] == "certain"]["sample_id"].tolist()

    if len(uncertain_ids) < 2 or len(certain_ids) < 2:
        log.warning(
            "sae_features: tiny splits (n_uncertain=%d, n_certain=%d); "
            "Cohen's d will be noisy",
            len(uncertain_ids), len(certain_ids),
        )

    vuf_meta = ctx.store.load_parquet("vuf/meta.parquet")
    available = sorted(int(x) for x in vuf_meta["layer"].tolist())
    target_layers = _resolve_layers(
        intervene_cfg.layer, available, model_name=ctx.cfg.model.name
    )

    hs_meta = ctx.store.load_parquet("hidden_states/meta.parquet").set_index("sample_id")
    k = min(int(sae_feat_cfg.k_top), ctx.sae.d_latent)
    rows: list[dict] = []

    for target_layer in target_layers:
        tensors = ctx.store.load_safetensors(f"hidden_states/layer_{target_layer}.safetensors")

        # The SAE's encoder weights expect a fixed input dimensionality. With
        # the default `SAEConfig.d_in=8` (FakeSAE) on top of a real model
        # (e.g. Qwen d_model=896), the failure today is a confusing torch
        # matmul shape error deep inside `sae.encode`. Surface it up front.
        sample_tensor = next(iter(tensors.values()))
        d_model = int(sample_tensor.shape[-1])
        if ctx.sae.d_in != d_model:
            raise ValueError(
                f"SAE.d_in={ctx.sae.d_in} != model hidden size d_model={d_model}; "
                f"either use provider=sae_lens (which infers d_in from the SAE) "
                f"or set sae.d_in to {d_model} in the config."
            )

        def pooled(sid: str, _tensors=tensors) -> "torch.Tensor":
            return _pool(
                _tensors[sid], vuf_cfg.pooling,
                int(hs_meta.loc[sid, "question_len"]),
                int(hs_meta.loc[sid, "seq_len"]),
            )

        X_u = torch.stack([pooled(sid) for sid in uncertain_ids]).float()
        X_c = torch.stack([pooled(sid) for sid in certain_ids]).float()

        log.info(
            "encoding %d uncertain + %d certain samples through SAE (layer %d, d_in=%d, "
            "d_latent=%d); k_top=%d",
            len(uncertain_ids), len(certain_ids), target_layer,
            ctx.sae.d_in, ctx.sae.d_latent, k,
        )

        f_u = ctx.sae.encode(X_u)
        f_c = ctx.sae.encode(X_c)
        d = _cohens_d(f_u, f_c)

        if sae_feat_cfg.selection_mode == "consensus":
            f_u_np = f_u.detach().cpu().numpy().astype(np.float32)
            f_c_np = f_c.detach().cpu().numpy().astype(np.float32)
            d_np = d.detach().cpu().numpy().astype(np.float64)
            uncertainty_idx, certainty_idx = _consensus_select(
                f_u_np, f_c_np, d_np,
                sae_feat_cfg=sae_feat_cfg, seed=int(ctx.cfg.seed), layer=target_layer,
            )
            log.info(
                "sae_features layer=%d (consensus): %d uncertainty, %d certainty "
                "features after bootstrap+FDR+|d| filter",
                target_layer, len(uncertainty_idx), len(certainty_idx),
            )
        else:
            # Keep |d|-largest with positive d as "uncertainty" features,
            # |d|-largest with negative d as "certainty" features. Both top-k.
            order_desc = d.argsort(descending=True).tolist()
            order_asc = d.argsort(descending=False).tolist()
            uncertainty_idx = set(order_desc[:k])
            certainty_idx = set(order_asc[:k])
        overlap = uncertainty_idx & certainty_idx
        if overlap:
            certainty_idx -= overlap
            log.warning(
                "sae_features layer=%d: %d feature ids were top-k in both "
                "directions; kept as uncertainty only",
                target_layer, len(overlap),
            )

        mean_u = f_u.mean(dim=0)
        mean_c = f_c.mean(dim=0)
        for i in range(ctx.sae.d_latent):
            if i in uncertainty_idx:
                selected = "uncertainty"
            elif i in certainty_idx:
                selected = "certainty"
            else:
                selected = ""
            rows.append({
                "feature_id": i,
                "layer": target_layer,
                "cohen_d": float(d[i].item()),
                "mean_uncertain": float(mean_u[i].item()),
                "mean_certain": float(mean_c[i].item()),
                "selected_as": selected,
            })

    ctx.store.save_parquet(OUTPUT, pd.DataFrame(rows))
    return [OUTPUT]
