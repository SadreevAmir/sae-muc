"""judge: ask the LLM-as-judge for a decisiveness score per generation.

Reads `samples.parquet` and `generations.parquet`, writes `judge_scores.parquet`
with one row per generation: `decisiveness` is the raw number from the judge,
and `vu_score = 1 - decisiveness`. Unparseable responses are left as None so
downstream code can decide how to handle them.
"""

from __future__ import annotations

import itertools
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from sae_muc.data.prompts import format_vu_judge_prompt
from sae_muc.pipeline._utils import select_vu_judge_rows
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
    concurrency = ctx.cfg.judge.concurrency
    log.info(
        "scoring VU for %d generations via %s (provider=%s, concurrency=%d) -> %s",
        total, ctx.cfg.judge.model, ctx.cfg.judge.provider, concurrency, output_name,
    )
    progress_every = max(1, total // 10)

    has_prompt_kind = "prompt_kind" in gens.columns
    # Thread-safe progress: workers complete out of order, so count
    # finished calls rather than relying on the result index.
    progress_lock = threading.Lock()
    done = itertools.count(1)

    def score_one(gen_row) -> tuple[dict, str]:
        question = samples.loc[gen_row["sample_id"], "question"]
        prompt = format_vu_judge_prompt(question=question, answer=gen_row["text"])

        # Per-prompt isolation: a flaky judge provider shouldn't kill the stage.
        status = "ok"
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
                status = "unparsed"
        except Exception as e:  # noqa: BLE001
            log.warning(
                "judge: giving up on sample_id=%s gen_idx=%d after retries: %s: %s",
                gen_row["sample_id"], int(gen_row["gen_idx"]), type(e).__name__, e,
            )
            text = f"ERROR: {type(e).__name__}: {e}"
            d = None
            status = "errored"

        row = {
            "sample_id": gen_row["sample_id"],
            "kind": gen_row["kind"],
            "gen_idx": int(gen_row["gen_idx"]),
            "decisiveness": d,
            "vu_score": (1.0 - d) if d is not None else None,
            "raw": text,
        }
        if has_prompt_kind:
            row["prompt_kind"] = gen_row.get("prompt_kind")

        with progress_lock:
            n = next(done)
            if n % progress_every == 0 and n < total:
                log.info("  progress: %d/%d (%d%%)", n, total, n * 100 // total)
        return row, status

    gen_rows = [gen_row for _, gen_row in gens.iterrows()]
    # Continuous worker pool: `concurrency` calls stay in flight and the next
    # task starts the instant any worker frees up. `map` preserves input order,
    # so the output rows match `gens` exactly (byte-identical at concurrency=1).
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(score_one, gen_rows))

    rows = [r for r, _ in results]
    unparsed = sum(1 for _, s in results if s == "unparsed")
    errored = sum(1 for _, s in results if s == "errored")

    if unparsed:
        log.warning("judge: %d/%d responses were unparseable", unparsed, len(rows))
    if errored:
        log.warning("judge: %d/%d responses errored after retries", errored, len(rows))

    ctx.store.save_parquet(output_name, pd.DataFrame(rows))
    return [output_name]


def run(ctx: PipelineContext) -> list[str]:
    gens = ctx.store.load_parquet("generations.parquet")
    # VU is measured on the eliciting samples + the plain most-likely answer
    # (paper §2.2/§2.3/§3.1); skip the plain samples / eliciting greedy that no
    # downstream consumer needs. eliciting_only runs judge the whole frame.
    return score_generations(ctx, select_vu_judge_rows(gens), OUTPUT)
