"""Unit tests for the prompt-regime row selectors (paper App A.1 / App C).

These pin down which rows each stage consumes under the plain/eliciting split:
SE + accuracy read the plain set, VU reads the eliciting samples + plain
greedy, VUF reads the eliciting set — and everything falls through unchanged
when there is no `prompt_kind` column (eliciting_only / steered generations).
"""

from __future__ import annotations

import pandas as pd

from sae_muc.pipeline._utils import (
    PROMPT_ELICITING,
    PROMPT_PLAIN,
    select_prompt_kind,
    select_vu_judge_rows,
)


def _split_gens() -> pd.DataFrame:
    rows = []
    for pk in (PROMPT_PLAIN, PROMPT_ELICITING):
        rows.append({"sample_id": "s0", "kind": "greedy", "gen_idx": 0, "prompt_kind": pk})
        for j in range(3):
            rows.append(
                {"sample_id": "s0", "kind": "sample", "gen_idx": j, "prompt_kind": pk}
            )
    return pd.DataFrame(rows)


def test_select_prompt_kind_filters_when_present():
    gens = _split_gens()
    plain = select_prompt_kind(gens, PROMPT_PLAIN)
    assert (plain["prompt_kind"] == PROMPT_PLAIN).all()
    assert len(plain) == 4  # 1 greedy + 3 samples


def test_select_prompt_kind_falls_through_without_column():
    # No prompt_kind column (eliciting_only / steered single set) -> pass-through.
    gens = pd.DataFrame(
        [{"sample_id": "s0", "kind": "greedy", "gen_idx": 0}]
    )
    out = select_prompt_kind(gens, PROMPT_PLAIN)
    assert out.equals(gens)


def test_select_prompt_kind_falls_through_when_value_absent():
    # Column exists but only eliciting rows -> asking for plain returns all.
    gens = _split_gens()
    elic_only = gens[gens["prompt_kind"] == PROMPT_ELICITING]
    out = select_prompt_kind(elic_only, PROMPT_PLAIN)
    assert out.equals(elic_only)


def test_select_vu_judge_rows_split():
    gens = _split_gens()
    rows = select_vu_judge_rows(gens)
    # Exactly the eliciting samples (3) + the plain greedy (1).
    assert len(rows) == 4
    elic = rows[rows["prompt_kind"] == PROMPT_ELICITING]
    plain = rows[rows["prompt_kind"] == PROMPT_PLAIN]
    assert (elic["kind"] == "sample").all() and len(elic) == 3
    assert (plain["kind"] == "greedy").all() and len(plain) == 1


def test_select_vu_judge_rows_eliciting_only_passthrough():
    # eliciting_only: no plain rows -> judge the whole frame (old behaviour).
    gens = _split_gens()
    elic_only = gens[gens["prompt_kind"] == PROMPT_ELICITING]
    out = select_vu_judge_rows(elic_only)
    assert out.equals(elic_only)
