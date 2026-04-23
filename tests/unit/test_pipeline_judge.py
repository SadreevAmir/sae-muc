from __future__ import annotations

import pytest

from sae_muc.models import Generation
from sae_muc.pipeline import generate, judge, prepare
from sae_muc.pipeline.judge import parse_decisiveness


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.8", 0.8),
        ("  0.5  ", 0.5),
        ("1.0", 1.0),
        ("0", 0.0),
        ("1", 1.0),
        (".6", 0.6),
        (" Decisiveness score: 0.4 (hedged)", 0.4),
        ("not a number", None),
        ("", None),
        ("2.0", None),        # out-of-range → None, not silently truncated
        ("-0.3", None),        # negatives are invalid
        ("3.5 then 0.8", 0.8),  # skip out-of-range, take next valid
    ],
)
def test_parse_decisiveness(text, expected):
    assert parse_decisiveness(text) == expected


def _patch_judge(ctx, text: str, *, monkeypatch):
    def fake_generate(prompts, **_kw):
        return [[Generation(text=text, finish_reason="stop")] for _ in prompts]

    monkeypatch.setattr(ctx.judge, "generate", fake_generate)


def test_judge_writes_parquet_with_scores(fake_ctx, monkeypatch):
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    _patch_judge(fake_ctx, "0.7", monkeypatch=monkeypatch)

    outputs = judge.run(fake_ctx)
    assert outputs == ["judge_scores.parquet"]

    df = fake_ctx.store.load_parquet("judge_scores.parquet")
    gens = fake_ctx.store.load_parquet("generations.parquet")
    # Every generation gets scored.
    assert len(df) == len(gens)
    assert (df["decisiveness"] == 0.7).all()
    # vu = 1 - decisiveness; tolerate float formatting.
    assert all(abs(v - 0.3) < 1e-9 for v in df["vu_score"])


def test_judge_tolerates_unparseable_responses(fake_ctx, monkeypatch):
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    _patch_judge(fake_ctx, "mumbo jumbo no number here", monkeypatch=monkeypatch)

    judge.run(fake_ctx)
    df = fake_ctx.store.load_parquet("judge_scores.parquet")
    assert df["decisiveness"].isna().all()
    assert df["vu_score"].isna().all()
    # Raw text is preserved for later inspection.
    assert (df["raw"] == "mumbo jumbo no number here").all()
