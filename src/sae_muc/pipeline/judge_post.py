"""judge_post: re-score VU on every intervention variant's generations.

For each row in `intervention/meta.parquet` (one per fixed-α value or a
single row for adaptive), read the intervened `generations.parquet`, run
the VU judge on it, and write `judge_scores.parquet` next to it inside
the same variant sub-directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sae_muc.pipeline import judge
from sae_muc.pipeline.context import PipelineContext

log = logging.getLogger(__name__)


def run(ctx: PipelineContext) -> list[str]:
    meta = ctx.store.load_parquet("intervention/meta.parquet")
    log.info("re-scoring VU for %d intervention variant(s)", len(meta))
    outputs: list[str] = []
    for _, row in meta.iterrows():
        src = str(row["path"])
        dst = str(Path(src).parent / "judge_scores.parquet")
        gens = ctx.store.load_parquet(src)
        outputs.extend(judge.score_generations(ctx, gens, dst))
    return outputs
