"""accuracy_judge: LLM-as-judge accuracy label for the greedy answer.

For each question we ask the judge whether the greedy answer is
semantically equivalent to any of the golden references (paper Appendix
A.3). The parsed boolean is stored in `accuracy.parquet`; unparseable
responses leave `is_correct` as None so downstream code can decide how
to handle them.
"""

from __future__ import annotations

import itertools
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from sae_muc.data.prompts import format_accuracy_judge_prompt
from sae_muc.pipeline._utils import PROMPT_PLAIN, select_prompt_kind
from sae_muc.pipeline.context import PipelineContext

log = logging.getLogger(__name__)

OUTPUT = "accuracy.parquet"

_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_RE = re.compile(r"\bno\b", re.IGNORECASE)


def parse_yes_no(text: str) -> bool | None:
    t = text.strip()
    if not t:
        return None
    # Anchor preference at the start of the response; fall back to anywhere.
    stripped = t.lstrip()
    if stripped.lower().startswith("yes"):
        return True
    if stripped.lower().startswith("no"):
        return False
    yes = _YES_RE.search(t)
    no = _NO_RE.search(t)
    if yes and not no:
        return True
    if no and not yes:
        return False
    if yes and no:
        # Both present — go by which appears first.
        return yes.start() < no.start()
    return None


def score_generations(
    ctx: PipelineContext,
    gens: pd.DataFrame,
    output_name: str,
) -> list[str]:
    """Ask the accuracy judge about the greedy rows in `gens`; save to `output_name`."""
    samples = ctx.store.load_parquet("samples.parquet").set_index("sample_id")
    # Accuracy is judged on the plain most-likely answer (paper App C / A.3:
    # the single low-T sequence off the plain question). Falls through to the
    # only greedy in eliciting_only / steered sets.
    greedy = select_prompt_kind(gens[gens["kind"] == "greedy"], PROMPT_PLAIN)
    total = len(greedy)
    concurrency = ctx.cfg.judge.concurrency
    log.info(
        "checking accuracy of %d greedy answers via %s (concurrency=%d) -> %s",
        total, ctx.cfg.judge.model, concurrency, output_name,
    )
    progress_every = max(1, total // 5)

    # Thread-safe progress: workers complete out of order, so count
    # finished calls rather than relying on the result index.
    progress_lock = threading.Lock()
    done = itertools.count(1)

    def score_one(greedy_row) -> tuple[dict, str]:
        sid = greedy_row["sample_id"]
        sample_row = samples.loc[sid]
        prompt = format_accuracy_judge_prompt(
            question=sample_row["question"],
            golden_answers=list(sample_row["gold_answers"]),
            answer=greedy_row["text"],
        )
        status = "ok"
        try:
            resp = ctx.judge.generate(
                [prompt],
                temperature=0.1,
                max_new_tokens=8,
                n=1,
            )
            raw = resp[0][0].text
            is_correct = parse_yes_no(raw)
            if is_correct is None:
                status = "unparsed"
        except Exception as e:  # noqa: BLE001
            log.warning(
                "accuracy_judge: giving up on sample_id=%s after retries: %s: %s",
                sid, type(e).__name__, e,
            )
            raw = f"ERROR: {type(e).__name__}: {e}"
            is_correct = None
            status = "errored"

        row = {"sample_id": sid, "is_correct": is_correct, "raw": raw}
        with progress_lock:
            n = next(done)
            if n % progress_every == 0 and n < total:
                log.info("  progress: %d/%d (%d%%)", n, total, n * 100 // total)
        return row, status

    greedy_rows = [r for _, r in greedy.iterrows()]
    # Continuous worker pool: `concurrency` calls stay in flight and the next
    # task starts the instant any worker frees up. `map` preserves input order,
    # so the output rows match `greedy` exactly (byte-identical at concurrency=1).
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(score_one, greedy_rows))

    rows = [r for r, _ in results]
    unparsed = sum(1 for _, s in results if s == "unparsed")
    errored = sum(1 for _, s in results if s == "errored")

    if unparsed:
        log.warning("accuracy_judge: %d/%d responses were unparseable", unparsed, len(rows))
    if errored:
        log.warning("accuracy_judge: %d/%d responses errored after retries", errored, len(rows))

    ctx.store.save_parquet(output_name, pd.DataFrame(rows))
    return [output_name]


def run(ctx: PipelineContext) -> list[str]:
    gens = ctx.store.load_parquet("generations.parquet")
    return score_generations(ctx, gens, OUTPUT)
