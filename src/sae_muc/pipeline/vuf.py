"""vuf: difference-in-means Verbal Uncertainty Feature, per layer.

Per paper §3.1: for each layer l,
    r_VU^(l) = mean(h(x)) over top-N uncertain questions
             - mean(h(x)) over bottom-N certain questions
then L2-normalised.

Per-question VU is the mean of judge scores over the N high-T sampled
generations (paper §2.2). The hidden state h(x) is pooled from the
stored full-sequence residual-stream tensor using the pooling mode from
config (default `last_token_q` — the paper's protocol).

We iterate over transformer layers only (layer_0 … layer_{n-1}); the
embedding output is available in `hidden_states/embedding.safetensors`
but is skipped by default for VUF.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from sae_muc.pipeline.context import PipelineContext

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

OUTPUT_META = "vuf/meta.parquet"
OUTPUT_SPLITS = "vuf/splits.parquet"


def _layer_in_path(layer: int) -> str:
    return f"hidden_states/layer_{layer}.safetensors"


def _direction_out_path(layer: int) -> str:
    return f"vuf/direction_layer_{layer}.safetensors"


def _pool(hs: "torch.Tensor", pooling: str, q_len: int, seq_len: int) -> "torch.Tensor":
    """hs shape: [seq_len, d_model]. Returns [d_model]."""
    q_len = max(1, min(q_len, seq_len))
    if pooling == "last_token_q":
        return hs[q_len - 1]
    if pooling == "last_token_a":
        return hs[-1]
    if pooling == "mean_q":
        return hs[:q_len].mean(dim=0)
    if pooling == "mean_a":
        if q_len >= seq_len:
            # Degenerate: no answer tokens. Fall back to the last question token.
            return hs[q_len - 1]
        return hs[q_len:].mean(dim=0)
    raise ValueError(f"Unknown pooling: {pooling!r}")


def _per_question_mean_vu(judge_df: pd.DataFrame) -> pd.Series:
    sampled = judge_df[judge_df["kind"] == "sample"]
    return sampled.groupby("sample_id")["vu_score"].mean().dropna()


def _resolve_layers(layers_cfg: list[int] | str, n_layers: int) -> list[int]:
    if layers_cfg == "auto":
        return list(range(n_layers))
    return [int(x) for x in layers_cfg]


def _split_ids(vu_per_q: pd.Series, n_top: int, n_bot: int) -> tuple[list[str], list[str]]:
    """Top-N uncertain + bottom-N certain, clamped so the two sets don't overlap."""
    total = len(vu_per_q)
    n_top = min(max(n_top, 1), total)
    n_bot = min(max(n_bot, 1), total)
    if n_top + n_bot > total:
        n_top = total // 2
        n_bot = total - n_top
    sorted_ids = vu_per_q.sort_values(ascending=False)
    return (
        sorted_ids.iloc[:n_top].index.tolist(),
        sorted_ids.iloc[total - n_bot :].index.tolist(),
    )


def run(ctx: PipelineContext) -> list[str]:
    import torch

    stage_cfg = ctx.cfg.stages.vuf

    judge = ctx.store.load_parquet("judge_scores.parquet")
    vu_per_q = _per_question_mean_vu(judge)
    uncertain_ids, certain_ids = _split_ids(vu_per_q, stage_cfg.n_top, stage_cfg.n_bot)

    # Persist split for downstream stages (sae_features) to reuse without
    # recomputing the sort. One row per question with the VU that drove it.
    splits_rows: list[dict] = []
    for sid, vu in vu_per_q.items():
        if sid in uncertain_ids:
            split = "uncertain"
        elif sid in certain_ids:
            split = "certain"
        else:
            split = "middle"
        splits_rows.append({"sample_id": sid, "mean_vu": float(vu), "split": split})
    ctx.store.save_parquet(OUTPUT_SPLITS, pd.DataFrame(splits_rows))

    meta = ctx.store.load_parquet("hidden_states/meta.parquet").set_index("sample_id")
    n_layers = int(meta.iloc[0]["n_layers"])
    layers = _resolve_layers(stage_cfg.layers, n_layers)
    log.info(
        "diff-in-means VUF: top %d uncertain vs bottom %d certain (pooling=%s, layers=%d)",
        len(uncertain_ids), len(certain_ids), stage_cfg.pooling, len(layers),
    )

    outputs: list[str] = []
    dir_meta: list[dict] = []

    for layer in layers:
        tensors = ctx.store.load_safetensors(_layer_in_path(layer))

        def pooled(sid: str) -> "torch.Tensor":
            return _pool(
                tensors[sid],
                stage_cfg.pooling,
                int(meta.loc[sid, "question_len"]),
                int(meta.loc[sid, "seq_len"]),
            )

        uncertain = torch.stack([pooled(sid) for sid in uncertain_ids])
        certain = torch.stack([pooled(sid) for sid in certain_ids])

        raw = uncertain.mean(dim=0) - certain.mean(dim=0)
        norm = raw.norm().item()
        direction = raw / norm if norm > 0 else raw

        path = _direction_out_path(layer)
        ctx.store.save_safetensors(path, {"direction": direction.contiguous()})
        outputs.append(path)

        dir_meta.append(
            {
                "layer": layer,
                "path": path,
                "raw_norm": float(norm),
                "n_uncertain": len(uncertain_ids),
                "n_certain": len(certain_ids),
                "pooling": stage_cfg.pooling,
            }
        )

    ctx.store.save_parquet(OUTPUT_META, pd.DataFrame(dir_meta))
    outputs.append(OUTPUT_META)
    outputs.append(OUTPUT_SPLITS)
    return outputs
