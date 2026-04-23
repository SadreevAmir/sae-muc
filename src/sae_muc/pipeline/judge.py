"""judge: ask the LLM-as-judge for a decisiveness score per generation.

Reads `samples.parquet` and `generations.parquet`, writes `judge_scores.parquet`
with one row per generation: `decisiveness` is the raw number from the judge,
and `vu_score = 1 - decisiveness`. Unparseable responses are left as None so
downstream code can decide how to handle them.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from sae_muc.data.prompts import format_vu_judge_prompt
from sae_muc.pipeline.context import PipelineContext

log = logging.getLogger(__name__)

INPUTS = ("samples.parquet", "generations.parquet")
OUTPUT = "judge_scores.parquet"

# Match any decimal or integer, optionally signed. We scan left-to-right
# and return the first value that actually falls in [0, 1]; this way "2.0"
# is rejected (not silently truncated to 0.0 from ".0").
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?|-?\.\d+")


def parse_decisiveness(text: str) -> float | None:
    for match in _NUMBER_RE.finditer(text):
        try:
            val = float(match.group(0))
        except ValueError:
            continue
        if 0.0 <= val <= 1.0:
            return val
    return None


def run(ctx: PipelineContext) -> list[str]:
    samples = ctx.store.load_parquet("samples.parquet").set_index("sample_id")
    gens = ctx.store.load_parquet("generations.parquet")

    prompts: list[str] = []
    for _, row in gens.iterrows():
        question = samples.loc[row["sample_id"], "question"]
        prompts.append(format_vu_judge_prompt(question=question, answer=row["text"]))

    responses = ctx.judge.generate(
        prompts,
        temperature=0.1,
        max_new_tokens=16,
        n=1,
    )

    rows: list[dict] = []
    unparsed = 0
    for (_, gen_row), resp in zip(gens.iterrows(), responses, strict=True):
        text = resp[0].text
        d = parse_decisiveness(text)
        if d is None:
            unparsed += 1
        rows.append(
            {
                "sample_id": gen_row["sample_id"],
                "kind": gen_row["kind"],
                "gen_idx": gen_row["gen_idx"],
                "decisiveness": d,
                "vu_score": (1.0 - d) if d is not None else None,
                "raw": text,
            }
        )

    if unparsed:
        log.warning("judge: %d/%d responses were unparseable", unparsed, len(rows))

    ctx.store.save_parquet(OUTPUT, pd.DataFrame(rows))
    return [OUTPUT]
