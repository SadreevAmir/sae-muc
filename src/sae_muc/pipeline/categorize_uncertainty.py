"""categorize_uncertainty: split the uncertain bucket into ABSTAIN vs HEDGE.

Project extension on top of the paper. The paper's decisiveness judge can't
distinguish "I don't know" (ABSTAIN) from "I think X" (HEDGE) — both push
decisiveness toward 0 and land in the same `uncertain = {VU ≥ 0.9}` set
that the linear VUF averages over. This stage runs a second LLM-as-judge
pass with a classification prompt so `vuf.run` can build per-category
directions and the supervisor's disentanglement question gets a scalar
answer in `diagnostics/category_directions.parquet`.

Behaviour:
  * Opt-in via `cfg.stages.categorize.enabled`. Default `False` writes
    an empty (zero-row) `categories.parquet` so the manifest is happy and
    `vuf` branches cleanly on file *contents*.
  * Reuses `ctx.judge` — same OpenRouter / CherryIn backend already wired.
  * Only categorises the uncertain bucket (`mean_vu ≥ cfg.stages.vuf.vu_uncertain_min`
    over the N high-T samples). Certain-bucket questions are skipped —
    labels there are meaningless, and we cut judge cost ~50%.
  * Per-generation classification → per-question majority vote (≥60% of
    valid votes, else `MIXED`). CONFIDENT-labelled generations are kept
    in the per-gen rows but excluded from the vote (treated as
    noise floor — judge said this gen doesn't actually look uncertain).

Output: `categories.parquet` with two row types (`level`):
  * `level=="gen"`     — one row per (sample_id, gen_idx), with `category`,
                         `raw` (judge response verbatim).
  * `level=="question"`— one row per sample_id, with aggregate `category`
                         and vote counts (`n_abstain`, `n_hedge`,
                         `n_confident`, `n_unparsed`).
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from sae_muc.data.prompts import format_categorize_prompt
from sae_muc.pipeline._utils import PROMPT_ELICITING, main_sample_ids, select_prompt_kind
from sae_muc.pipeline.context import PipelineContext

log = logging.getLogger(__name__)

OUTPUT = "categories.parquet"

# Whole-word, case-insensitive — anchors against judge prefixes
# ("The category is ABSTAIN.", "Category: HEDGE\n"). First match wins.
_CATEGORY_RE = re.compile(r"\b(ABSTAIN|HEDGE|CONFIDENT)\b", re.IGNORECASE)

# Categories the judge can return. MIXED / None are aggregation artifacts,
# not labels the judge produces directly.
_VALID_CATEGORIES = ("ABSTAIN", "HEDGE", "CONFIDENT")

# Aggregation: a non-MIXED label requires this fraction of *valid* votes.
# Valid = ABSTAIN or HEDGE (CONFIDENT excluded from the denominator since
# the question is "abstain vs hedge among uncertain answers").
_MAJORITY_THRESHOLD = 0.6


def parse_category(text: str) -> str | None:
    """Extract the first whole-word ABSTAIN / HEDGE / CONFIDENT (case-insensitive)."""
    if not text:
        return None
    m = _CATEGORY_RE.search(text)
    if m is None:
        return None
    return m.group(1).upper()


def aggregate_votes(labels: list[str | None]) -> tuple[str, dict[str, int]]:
    """Per-question majority vote.

    Returns (aggregate_label, counts_dict). Counts include n_unparsed.
    Aggregate is one of ABSTAIN / HEDGE / MIXED. CONFIDENT generations
    are tallied but excluded from the vote denominator: the question
    "abstain vs hedge among uncertain answers" makes sense only for
    generations that *are* uncertain.
    """
    counts = {"ABSTAIN": 0, "HEDGE": 0, "CONFIDENT": 0, "unparsed": 0}
    for lbl in labels:
        if lbl is None:
            counts["unparsed"] += 1
        elif lbl in counts:
            counts[lbl] += 1
    valid = counts["ABSTAIN"] + counts["HEDGE"]
    if valid == 0:
        return "MIXED", counts
    # Threshold scales with the actual valid-vote count (not the total
    # generation count) so a question with 3/10 CONFIDENT + 4 ABSTAIN +
    # 3 HEDGE doesn't get rejected just because CONFIDENT diluted things.
    for label in ("ABSTAIN", "HEDGE"):
        if counts[label] / valid >= _MAJORITY_THRESHOLD:
            return label, counts
    return "MIXED", counts


def _per_question_mean_vu(judge_df: pd.DataFrame) -> pd.Series:
    """Mean VU over the N high-T samples per question (mirrors vuf.py)."""
    sampled = judge_df[judge_df["kind"] == "sample"]
    return sampled.groupby("sample_id")["vu_score"].mean().dropna()


def _empty_stub(ctx: PipelineContext) -> list[str]:
    """Write a zero-row valid parquet so the manifest + downstream branching
    behave consistently across enabled=False ↔ True re-runs."""
    df = pd.DataFrame(
        {
            "level": pd.Series([], dtype="object"),
            "sample_id": pd.Series([], dtype="object"),
            "gen_idx": pd.Series([], dtype="int64"),
            "category": pd.Series([], dtype="object"),
            "raw": pd.Series([], dtype="object"),
            "n_abstain": pd.Series([], dtype="int64"),
            "n_hedge": pd.Series([], dtype="int64"),
            "n_confident": pd.Series([], dtype="int64"),
            "n_unparsed": pd.Series([], dtype="int64"),
        }
    )
    ctx.store.save_parquet(OUTPUT, df)
    return [OUTPUT]


def run(ctx: PipelineContext) -> list[str]:
    cat_cfg = ctx.cfg.stages.categorize
    if not cat_cfg.enabled:
        log.info("categorize_uncertainty: disabled by config; writing empty stub")
        return _empty_stub(ctx)

    judge_df = ctx.store.load_parquet("judge_scores.parquet")
    gens = ctx.store.load_parquet("generations.parquet")
    samples = ctx.store.load_parquet("samples.parquet").set_index("sample_id")

    vu_per_q = _per_question_mean_vu(judge_df)
    # Categorisation is fit-prep for per-category VUF directions, so restrict it
    # to the main split — held-out questions must stay out of every fit.
    vu_per_q = vu_per_q[vu_per_q.index.isin(main_sample_ids(ctx))]
    threshold = float(ctx.cfg.stages.vuf.vu_uncertain_min)
    uncertain_ids = set(vu_per_q.index[vu_per_q >= threshold].tolist())

    # The uncertain bucket is defined on eliciting-prompt VU, so categorise the
    # eliciting samples (falls through in eliciting_only mode).
    sample_gens = select_prompt_kind(gens[gens["kind"] == "sample"], PROMPT_ELICITING)
    candidates = sample_gens[sample_gens["sample_id"].isin(uncertain_ids)]
    log.info(
        "categorize_uncertainty: %d generations to label (%d uncertain questions × ≤N samples)",
        len(candidates), len(uncertain_ids),
    )

    gen_rows: list[dict[str, Any]] = []
    progress_every = max(1, len(candidates) // 10)

    for i, (_, gen_row) in enumerate(candidates.iterrows()):
        sid = str(gen_row["sample_id"])
        gen_idx = int(gen_row["gen_idx"])
        question = str(samples.loc[sid, "question"])
        answer = str(gen_row["text"])
        prompt = format_categorize_prompt(question, answer)
        try:
            resp = ctx.judge.generate(
                [prompt], temperature=0.0, max_new_tokens=8, n=1,
            )
            raw = resp[0][0].text
            category = parse_category(raw)
        except Exception as e:  # noqa: BLE001
            log.warning("categorize: judge call failed on %s gen=%d (%s)", sid, gen_idx, e)
            raw, category = "", None

        gen_rows.append(
            {
                "level": "gen",
                "sample_id": sid,
                "gen_idx": gen_idx,
                "category": category,
                "raw": raw,
                "n_abstain": 0,
                "n_hedge": 0,
                "n_confident": 0,
                "n_unparsed": 0,
            }
        )
        if (i + 1) % progress_every == 0:
            log.info("categorize: %d/%d", i + 1, len(candidates))

    # Aggregate per question.
    by_sid: dict[str, list[str | None]] = {}
    for r in gen_rows:
        by_sid.setdefault(r["sample_id"], []).append(r["category"])

    agg_rows: list[dict[str, Any]] = []
    for sid in sorted(by_sid):
        label, counts = aggregate_votes(by_sid[sid])
        agg_rows.append(
            {
                "level": "question",
                "sample_id": sid,
                "gen_idx": -1,
                "category": label,
                "raw": "",
                "n_abstain": counts["ABSTAIN"],
                "n_hedge": counts["HEDGE"],
                "n_confident": counts["CONFIDENT"],
                "n_unparsed": counts["unparsed"],
            }
        )

    log.info(
        "categorize_uncertainty: aggregate counts ABSTAIN=%d HEDGE=%d MIXED=%d (n=%d)",
        sum(1 for r in agg_rows if r["category"] == "ABSTAIN"),
        sum(1 for r in agg_rows if r["category"] == "HEDGE"),
        sum(1 for r in agg_rows if r["category"] == "MIXED"),
        len(agg_rows),
    )

    df = pd.DataFrame(gen_rows + agg_rows)
    ctx.store.save_parquet(OUTPUT, df)
    return [OUTPUT]
