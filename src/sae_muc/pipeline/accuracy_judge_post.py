"""accuracy_judge_post: re-check greedy accuracy on every intervention variant."""

from __future__ import annotations

import logging
from pathlib import Path

from sae_muc.pipeline import accuracy_judge
from sae_muc.pipeline.context import PipelineContext

log = logging.getLogger(__name__)


def run(ctx: PipelineContext) -> list[str]:
    meta = ctx.store.load_parquet("intervention/meta.parquet")
    log.info("re-checking accuracy for %d intervention variant(s)", len(meta))
    outputs: list[str] = []
    for _, row in meta.iterrows():
        src = str(row["path"])
        dst = str(Path(src).parent / "accuracy.parquet")
        gens = ctx.store.load_parquet(src)
        outputs.extend(accuracy_judge.score_generations(ctx, gens, dst))
    return outputs
