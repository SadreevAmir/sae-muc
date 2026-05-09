"""semantic_entropy_post: re-cluster sampled answers for every intervention variant."""

from __future__ import annotations

import logging
from pathlib import Path

from sae_muc.pipeline import semantic_entropy
from sae_muc.pipeline.context import PipelineContext

log = logging.getLogger(__name__)


def run(ctx: PipelineContext) -> list[str]:
    meta = ctx.store.load_parquet("intervention/meta.parquet")
    log.info("re-clustering SE for %d intervention variant(s)", len(meta))
    outputs: list[str] = []
    for _, row in meta.iterrows():
        src = str(row["path"])
        dst = str(Path(src).parent / "semantic_entropy.parquet")
        gens = ctx.store.load_parquet(src)
        outputs.extend(semantic_entropy.cluster_generations(ctx, gens, dst))
    return outputs
