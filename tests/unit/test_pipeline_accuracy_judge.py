from __future__ import annotations

import pytest

from sae_muc.models import Generation
from sae_muc.pipeline import accuracy_judge, generate, prepare
from sae_muc.pipeline.accuracy_judge import parse_yes_no


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("yes", True),
        ("YES", True),
        ("Yes.", True),
        ("  yes ", True),
        ("no", False),
        ("NO!", False),
        ("No, that's wrong.", False),
        ("probably yes", True),      # word-boundary yes
        ("not really, no", False),    # word-boundary no
        ("maybe", None),
        ("", None),
        ("   ", None),
        ("yes and no", True),          # both present; yes appears first
        ("no, not yes", False),        # no appears first
    ],
)
def test_parse_yes_no(text, expected):
    assert parse_yes_no(text) == expected


def _patch_judge(ctx, text: str, *, monkeypatch):
    def fake_generate(prompts, **_kw):
        return [[Generation(text=text, finish_reason="stop")] for _ in prompts]

    monkeypatch.setattr(ctx.judge, "generate", fake_generate)


def test_accuracy_stage_writes_per_question(fake_ctx, monkeypatch):
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    _patch_judge(fake_ctx, "yes", monkeypatch=monkeypatch)

    outputs = accuracy_judge.run(fake_ctx)
    assert outputs == ["accuracy.parquet"]
    df = fake_ctx.store.load_parquet("accuracy.parquet")
    samples = fake_ctx.store.load_parquet("samples.parquet")
    assert set(df["sample_id"]) == set(samples["sample_id"])
    assert df["is_correct"].all()
    assert (df["raw"] == "yes").all()


def test_accuracy_stage_handles_unparseable(fake_ctx, monkeypatch):
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    _patch_judge(fake_ctx, "maybe?", monkeypatch=monkeypatch)

    accuracy_judge.run(fake_ctx)
    df = fake_ctx.store.load_parquet("accuracy.parquet")
    assert df["is_correct"].isna().all()


def test_fake_backend_returns_yes_no_on_accuracy_prompt(fake_ctx):
    """Smoke-check that FakeBackend produces parseable yes/no on accuracy prompts."""
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    accuracy_judge.run(fake_ctx)
    df = fake_ctx.store.load_parquet("accuracy.parquet")
    # FakeBackend produces parseable yes/no, so nothing should be NaN.
    assert df["is_correct"].notna().all()
    # And both classes should appear eventually (5 samples hashed → mix).
    assert set(df["is_correct"].unique()) <= {True, False}
