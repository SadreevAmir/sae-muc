"""prepare: load questions from the configured dataset and snapshot them to parquet.

Stages downstream read `samples.parquet` rather than hitting HuggingFace again,
so the pipeline is self-contained inside the run directory.
"""

from __future__ import annotations

import logging

import pandas as pd

from sae_muc.data import load_samples
from sae_muc.pipeline.context import PipelineContext

log = logging.getLogger(__name__)

OUTPUT = "samples.parquet"


def run(ctx: PipelineContext) -> list[str]:
    cfg = ctx.cfg.dataset
    log.info(
        "loading dataset %s (split=%s, n=%d, heldout=%d, seed=%d)",
        cfg.name, cfg.split, cfg.n_samples, cfg.heldout_n, cfg.seed,
    )
    samples = load_samples(ctx.cfg.dataset)
    df = pd.DataFrame(
        {
            "sample_id": [s.sample_id for s in samples],
            "question": [s.question for s in samples],
            "gold_answers": [list(s.gold_answers) for s in samples],
            "split": [s.split for s in samples],
        }
    )
    ctx.store.save_parquet(OUTPUT, df)
    return [OUTPUT]
