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


def score_generations(
    ctx: PipelineContext,
    gens: pd.DataFrame,
    output_name: str,
) -> list[str]:
    """Score VU for every row of `gens` and write the result at `output_name`.

    Used by both `run()` (baseline generations) and `judge_post.run()`
    (intervened generations).
    """
    samples = ctx.store.load_parquet("samples.parquet").set_index("sample_id")
    total = len(gens)
    log.info(
        "scoring VU for %d generations via %s (provider=%s) -> %s",
        total, ctx.cfg.judge.model, ctx.cfg.judge.provider, output_name,
    )
    progress_every = max(1, total // 10)

    rows: list[dict] = []
    unparsed = 0
    errored = 0
    for i, (_, gen_row) in enumerate(gens.iterrows()):
        question = samples.loc[gen_row["sample_id"], "question"]
        prompt = format_vu_judge_prompt(question=question, answer=gen_row["text"])

        # Per-prompt isolation: a flaky judge provider shouldn't kill the stage.
        try:
            resp = ctx.judge.generate(
                [prompt],
                temperature=0.1,
                max_new_tokens=16,
                n=1,
            )
            text = resp[0][0].text
            d = parse_decisiveness(text)
            if d is None:
                unparsed += 1
        except Exception as e:  # noqa: BLE001
            log.warning(
                "judge: giving up on sample_id=%s gen_idx=%d after retries: %s: %s",
                gen_row["sample_id"], int(gen_row["gen_idx"]), type(e).__name__, e,
            )
            text = f"ERROR: {type(e).__name__}: {e}"
            d = None
            errored += 1

        rows.append(
            {
                "sample_id": gen_row["sample_id"],
                "kind": gen_row["kind"],
                "gen_idx": int(gen_row["gen_idx"]),
                "decisiveness": d,
                "vu_score": (1.0 - d) if d is not None else None,
                "raw": text,
            }
        )
        if (i + 1) % progress_every == 0 and (i + 1) < total:
            log.info("  progress: %d/%d (%d%%)", i + 1, total, (i + 1) * 100 // total)

    if unparsed:
        log.warning("judge: %d/%d responses were unparseable", unparsed, len(rows))
    if errored:
        log.warning("judge: %d/%d responses errored after retries", errored, len(rows))

    ctx.store.save_parquet(output_name, pd.DataFrame(rows))
    return [output_name]


def run(ctx: PipelineContext) -> list[str]:
    gens = ctx.store.load_parquet("generations.parquet")
    return score_generations(ctx, gens, OUTPUT)
