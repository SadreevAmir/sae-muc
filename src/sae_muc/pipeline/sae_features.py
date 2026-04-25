"""sae_features: SAE-latent feature analysis for the uncertain / certain split.

Encode pooled hidden states (at the intervene layer) through the SAE,
then rank latent features by Cohen's d between the uncertain and certain
question sets. The top-k positive-d features become candidate
"uncertainty" features; the top-k negative-d features become "certainty"
features. Downstream intervene methods `sae_emd` and `sae_clamp` read
the resulting selection from `sae_features/stats.parquet`.

Required upstream stages: vuf (for `vuf/splits.parquet`,
`vuf/meta.parquet`) and hidden_states (for the per-sample layer tensor).
SAE comes from `ctx.sae`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from sae_muc.pipeline._utils import _pool, _resolve_layer
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
    target_layer = _resolve_layer(intervene_cfg.layer, available)

    meta = ctx.store.load_parquet("hidden_states/meta.parquet").set_index("sample_id")
    tensors = ctx.store.load_safetensors(f"hidden_states/layer_{target_layer}.safetensors")

    # The SAE's encoder weights expect a fixed input dimensionality. With the
    # default `SAEConfig.d_in=8` (FakeSAE) on top of a real model (e.g. Qwen
    # d_model=896), the failure today is a confusing torch matmul shape error
    # deep inside `sae.encode`. Surface it up front instead.
    sample_tensor = next(iter(tensors.values()))
    d_model = int(sample_tensor.shape[-1])
    if ctx.sae.d_in != d_model:
        raise ValueError(
            f"SAE.d_in={ctx.sae.d_in} != model hidden size d_model={d_model}; "
            f"either use provider=sae_lens (which infers d_in from the SAE) "
            f"or set sae.d_in to {d_model} in the config."
        )

    def pooled(sid: str) -> "torch.Tensor":
        return _pool(
            tensors[sid], vuf_cfg.pooling,
            int(meta.loc[sid, "question_len"]),
            int(meta.loc[sid, "seq_len"]),
        )

    X_u = torch.stack([pooled(sid) for sid in uncertain_ids]).float()
    X_c = torch.stack([pooled(sid) for sid in certain_ids]).float()

    log.info(
        "encoding %d uncertain + %d certain samples through SAE (layer %d, d_in=%d, "
        "d_latent=%d); k_top=%d",
        len(uncertain_ids), len(certain_ids), target_layer,
        ctx.sae.d_in, ctx.sae.d_latent, sae_feat_cfg.k_top,
    )

    f_u = ctx.sae.encode(X_u)
    f_c = ctx.sae.encode(X_c)
    d = _cohens_d(f_u, f_c)

    # Keep |d|-largest with positive d as "uncertainty" features, |d|-largest
    # with negative d as "certainty" features. Both top-k (k_top).
    k = int(sae_feat_cfg.k_top)
    k = min(k, ctx.sae.d_latent)
    order_desc = d.argsort(descending=True).tolist()
    order_asc = d.argsort(descending=False).tolist()
    uncertainty_idx = set(order_desc[:k])
    certainty_idx = set(order_asc[:k])
    # Clean potential overlap (shouldn't happen if k ≤ d_latent/2, but be safe).
    overlap = uncertainty_idx & certainty_idx
    if overlap:
        certainty_idx -= overlap
        log.warning(
            "sae_features: %d feature ids were top-k in both directions; kept as uncertainty only",
            len(overlap),
        )

    rows = []
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
