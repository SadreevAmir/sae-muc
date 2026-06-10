"""combine_vuf: pool per-dataset VUFs into one universal (or OOD) VUF.

Paper App G.1 p.29: "we combine the VUFs exacted from three datasets together
and construct D_certain and D_uncertain as samples with VU Score ≤ 0.05 and
≥ 0.9 respectively." That is a SINGLE diff-in-means over the *union* of the
three datasets' contrast sets, not an average of three precomputed VUFs.

We reconstruct that pooled diff-in-means exactly from the per-set mean
activations each source run saved (`vuf/means_layer_{L}.safetensors`) plus the
counts in its `vuf/meta.parquet`, without re-loading every activation:

    pooled_uncertain[l] = Σ_d n_unc_d · mean_uncertain_d[l] / Σ_d n_unc_d
    pooled_certain[l]   = Σ_d n_cert_d · mean_certain_d[l]   / Σ_d n_cert_d
    r_VU[l] = normalize(pooled_uncertain[l] − pooled_certain[l])     (Eq.2/Eq.3)

The result overwrites this run's `vuf/direction_layer_{L}.safetensors` +
`vuf/meta.parquet`, so the downstream `intervene` picks it up unchanged.

Modes (`cfg.stages.vuf.combine_sources`):
  * empty (default)  → no-op; the single-dataset VUF stands.
  * three sources    → the universal VUF (App G.1, headline Table-3 mitigation).
  * a single source  → OOD reuse: e.g. the TriviaQA VUF applied to NQ-Open /
                       PopQA (Table 5).

With ≥2 sources we also write `vuf/cross_dataset_cosine.parquet`: the per-layer
mean pairwise cosine similarity of the source VUFs — the quantitative half of
the §3.2 "effective layer selection" signal (high in the middle-to-last band).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from sae_muc.artifacts.store import ArtifactStore
from sae_muc.pipeline.context import PipelineContext

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

OUTPUT_META = "vuf/meta.parquet"
OUTPUT_COSINE = "vuf/cross_dataset_cosine.parquet"


def _means_path(layer: int) -> str:
    return f"vuf/means_layer_{layer}.safetensors"


def _direction_path(layer: int) -> str:
    return f"vuf/direction_layer_{layer}.safetensors"


def _resolve_source_dir(ctx: PipelineContext, source: str) -> Path:
    """A combine source is a run-dir path or a run_id under data_root/runs/."""
    p = Path(source)
    if (p / "vuf" / "meta.parquet").exists():
        return p
    candidate = Path(ctx.cfg.data_root) / "runs" / source
    if (candidate / "vuf" / "meta.parquet").exists():
        return candidate
    raise ValueError(
        f"combine_vuf source {source!r} not found: expected a run directory with "
        f"vuf/meta.parquet, or a run_id under {Path(ctx.cfg.data_root) / 'runs'}."
    )


def _mean_pairwise_cosine(dirs: list["torch.Tensor"]) -> float:
    import torch

    sims: list[float] = []
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            sims.append(float(torch.nn.functional.cosine_similarity(dirs[i], dirs[j], dim=0)))
    return float(sum(sims) / len(sims)) if sims else float("nan")


def run(ctx: PipelineContext) -> list[str]:
    import torch

    sources = list(ctx.cfg.stages.vuf.combine_sources)
    if not sources:
        log.info("combine_vuf: no combine_sources → single-dataset VUF stands (no-op)")
        return []

    stores = [ArtifactStore(_resolve_source_dir(ctx, s)) for s in sources]
    metas = [st.load_parquet("vuf/meta.parquet") for st in stores]

    # Pooling must be consistent across sources (and with this run) — pooling
    # means extracted under different token positions would yield a meaningless
    # mixed direction. Fail loud rather than silently combine apples and oranges.
    poolings = {str(m["pooling"].iloc[0]) for m in metas if len(m)}
    poolings.add(ctx.cfg.stages.vuf.pooling)
    if len(poolings) > 1:
        raise ValueError(
            f"combine_vuf: sources disagree on pooling ({sorted(poolings)}); "
            f"all source runs and this run must use the same vuf.pooling."
        )

    layer_sets = [set(int(x) for x in m["layer"].tolist()) for m in metas]
    common = sorted(set.intersection(*layer_sets))
    if not common:
        raise ValueError(
            f"combine_vuf: sources {sources} share no common VUF layers "
            f"(per-source layers: {[sorted(s) for s in layer_sets]})."
        )

    log.info(
        "combine_vuf: pooling %d source(s) over %d common layers → universal VUF",
        len(sources), len(common),
    )

    pooling = ctx.cfg.stages.vuf.pooling
    dir_meta: list[dict] = []
    cosine_rows: list[dict] = []

    for layer in common:
        unc_acc: torch.Tensor | None = None
        cert_acc: torch.Tensor | None = None
        n_unc_total = 0
        n_cert_total = 0
        source_dirs: list[torch.Tensor] = []
        for st, meta in zip(stores, metas, strict=True):
            means = st.load_safetensors(_means_path(layer))
            row = meta[meta["layer"] == layer].iloc[0]
            n_unc = int(row["n_uncertain"])
            n_cert = int(row["n_certain"])
            mu = means["mean_uncertain"].float()
            mc = means["mean_certain"].float()
            unc_acc = mu * n_unc if unc_acc is None else unc_acc + mu * n_unc
            cert_acc = mc * n_cert if cert_acc is None else cert_acc + mc * n_cert
            n_unc_total += n_unc
            n_cert_total += n_cert
            d = mu - mc
            dn = d.norm()
            source_dirs.append(d / dn if dn > 0 else d)

        pooled_uncertain = unc_acc / max(n_unc_total, 1)
        pooled_certain = cert_acc / max(n_cert_total, 1)
        raw = pooled_uncertain - pooled_certain
        norm = raw.norm().item()
        direction = raw / norm if norm > 0 else raw

        path = _direction_path(layer)
        ctx.store.save_safetensors(path, {"direction": direction.contiguous()})
        dir_meta.append(
            {
                "layer": layer,
                "path": path,
                "raw_norm": float(norm),
                "n_uncertain": n_unc_total,
                "n_certain": n_cert_total,
                "pooling": pooling,
            }
        )
        if len(source_dirs) >= 2:
            cosine_rows.append(
                {
                    "layer": layer,
                    "mean_cross_dataset_cosine": _mean_pairwise_cosine(source_dirs),
                    "n_sources": len(source_dirs),
                }
            )

    ctx.store.save_parquet(OUTPUT_META, pd.DataFrame(dir_meta))
    outputs = [_direction_path(l) for l in common] + [OUTPUT_META]
    if cosine_rows:
        ctx.store.save_parquet(OUTPUT_COSINE, pd.DataFrame(cosine_rows))
        outputs.append(OUTPUT_COSINE)
        log.info(
            "combine_vuf: cross-dataset cosine diagnostic written (%d layers)", len(cosine_rows)
        )
    return outputs
