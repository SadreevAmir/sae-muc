"""generate: greedy @ low-T plus N samples @ high-T for every question.

Reads `samples.parquet`, writes `generations.parquet` with one row per
generation. `kind` is `greedy` or `sample`; `gen_idx` enumerates samples
within a question (0 for the single greedy row).
"""

from __future__ import annotations

import logging

import pandas as pd

from sae_muc.data import format_answer_prompt
from sae_muc.pipeline.context import PipelineContext

log = logging.getLogger(__name__)

INPUT = "samples.parquet"
OUTPUT = "generations.parquet"


def run(ctx: PipelineContext) -> list[str]:
    stage_cfg = ctx.cfg.stages.generate
    samples = ctx.store.load_parquet(INPUT)

    prompts = [format_answer_prompt(q, eliciting=True) for q in samples["question"]]
    log.info(
        "generating 1 greedy @T=%.2f + %d samples @T=%.2f per question (%d questions, max_new_tokens=%d)",
        stage_cfg.temperature_low, stage_cfg.n_samples, stage_cfg.temperature_high,
        len(prompts), stage_cfg.max_new_tokens,
    )

    greedy = ctx.llm.generate(
        prompts,
        temperature=stage_cfg.temperature_low,
        max_new_tokens=stage_cfg.max_new_tokens,
        n=1,
        seed=ctx.cfg.seed,
    )
    sampled = ctx.llm.generate(
        prompts,
        temperature=stage_cfg.temperature_high,
        max_new_tokens=stage_cfg.max_new_tokens,
        n=stage_cfg.n_samples,
        seed=ctx.cfg.seed,
    )

    rows: list[dict] = []
    for sample_id, g_list, s_list in zip(samples["sample_id"], greedy, sampled, strict=True):
        rows.append(
            {
                "sample_id": sample_id,
                "kind": "greedy",
                "gen_idx": 0,
                "text": g_list[0].text,
                "finish_reason": g_list[0].finish_reason,
            }
        )
        for j, gen in enumerate(s_list):
            rows.append(
                {
                    "sample_id": sample_id,
                    "kind": "sample",
                    "gen_idx": j,
                    "text": gen.text,
                    "finish_reason": gen.finish_reason,
                }
            )

    ctx.store.save_parquet(OUTPUT, pd.DataFrame(rows))
    return [OUTPUT]
