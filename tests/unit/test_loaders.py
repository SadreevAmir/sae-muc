from __future__ import annotations

import json
import random

import pytest

from sae_muc.config import DatasetConfig
from sae_muc.data import load_samples


class _FakeDS:
    """Minimal drop-in for a HuggingFace `datasets.Dataset` used in loaders."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = list(rows)

    def shuffle(self, seed: int) -> "_FakeDS":
        rng = random.Random(seed)
        shuffled = list(self.rows)
        rng.shuffle(shuffled)
        return _FakeDS(shuffled)

    def select(self, indices) -> "_FakeDS":
        return _FakeDS([self.rows[i] for i in indices])

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)


def _patch_hf(monkeypatch, table: dict[str, list[dict]]) -> None:
    """Install a fake HF `load_dataset` that dispatches on repo name."""

    def fake_load(repo: str, *args, **kwargs):
        rows = table.get(repo)
        if rows is None:
            raise AssertionError(f"Unexpected HF dataset requested: {repo}")
        return _FakeDS(rows)

    monkeypatch.setattr("sae_muc.data.loaders._hf_load_dataset", fake_load)


def test_load_triviaqa_normalises_aliases(monkeypatch):
    _patch_hf(
        monkeypatch,
        {
            "mandarjoshi/trivia_qa": [
                {
                    "question": "Who wrote War and Peace?",
                    "answer": {"value": "Leo Tolstoy", "aliases": ["Tolstoy", "Leo Tolstoy"]},
                },
                {
                    "question": "Capital of France?",
                    "answer": {"value": "Paris", "aliases": []},
                },
            ]
        },
    )
    cfg = DatasetConfig(name="triviaqa", split="validation", n_samples=10, seed=0)
    samples = load_samples(cfg)
    assert len(samples) == 2
    first = {s.question: s for s in samples}
    assert first["Capital of France?"].gold_answers == ["Paris"]
    wt = first["Who wrote War and Peace?"]
    # De-duplicated, value-first order.
    assert wt.gold_answers == ["Leo Tolstoy", "Tolstoy"]
    assert wt.sample_id.startswith("triviaqa:validation:")


def _patch_parquet(monkeypatch, rows: list[dict]) -> None:
    """Install a fake `_hf_load_parquet` that returns a DataFrame of `rows`."""
    import pandas as pd

    df = pd.DataFrame(rows)
    monkeypatch.setattr("sae_muc.data.loaders._hf_load_parquet", lambda *a, **k: df)


def test_load_nq_open(monkeypatch):
    _patch_parquet(
        monkeypatch,
        [
            {"question": "Q1", "answer": ["a1", "a2"]},
            {"question": "Q2", "answer": ["b1"]},
        ],
    )
    cfg = DatasetConfig(name="nq_open", split="validation", n_samples=10, seed=0)
    samples = load_samples(cfg)
    assert {s.question for s in samples} == {"Q1", "Q2"}
    q1 = next(s for s in samples if s.question == "Q1")
    assert q1.gold_answers == ["a1", "a2"]


def test_load_popqa_parses_json_possible_answers(monkeypatch):
    _patch_hf(
        monkeypatch,
        {
            "akariasai/PopQA": [
                {
                    "question": "Who is X?",
                    "possible_answers": json.dumps(["foo", "bar"]),
                    "obj": "foo",
                },
                {
                    "question": "Who is Y?",
                    "possible_answers": ["baz"],
                    "obj": "baz",
                },
            ]
        },
    )
    cfg = DatasetConfig(name="popqa", split="test", n_samples=10, seed=0)
    samples = load_samples(cfg)
    golds = {s.question: s.gold_answers for s in samples}
    assert golds["Who is X?"] == ["foo", "bar"]
    assert golds["Who is Y?"] == ["baz"]


def test_n_samples_caps_output(monkeypatch):
    rows = [{"question": f"q{i}", "answer": [f"a{i}"]} for i in range(20)]
    _patch_parquet(monkeypatch, rows)

    cfg = DatasetConfig(name="nq_open", split="validation", n_samples=5, seed=42)
    samples = load_samples(cfg)
    assert len(samples) == 5


def test_seed_makes_sampling_deterministic(monkeypatch):
    rows = [{"question": f"q{i}", "answer": [f"a{i}"]} for i in range(20)]
    _patch_parquet(monkeypatch, rows)

    cfg_a = DatasetConfig(name="nq_open", split="validation", n_samples=5, seed=42)
    cfg_b = DatasetConfig(name="nq_open", split="validation", n_samples=5, seed=42)
    cfg_c = DatasetConfig(name="nq_open", split="validation", n_samples=5, seed=999)

    a = [s.question for s in load_samples(cfg_a)]
    b = [s.question for s in load_samples(cfg_b)]
    c = [s.question for s in load_samples(cfg_c)]
    assert a == b
    assert a != c


def test_unknown_dataset_raises():
    # Bypass pydantic enum by constructing via model_construct.
    cfg = DatasetConfig.model_construct(name="mystery", split="x", n_samples=1, seed=0)
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_samples(cfg)


def test_fake_dataset_is_offline_and_deterministic():
    """`name=fake` must not call the HF loader and must be deterministic from cfg.seed."""
    cfg_a = DatasetConfig(name="fake", split="validation", n_samples=4, seed=0)
    cfg_b = DatasetConfig(name="fake", split="validation", n_samples=4, seed=0)
    cfg_c = DatasetConfig(name="fake", split="validation", n_samples=4, seed=7)

    a = [(s.sample_id, s.question, tuple(s.gold_answers)) for s in load_samples(cfg_a)]
    b = [(s.sample_id, s.question, tuple(s.gold_answers)) for s in load_samples(cfg_b)]
    c = [(s.sample_id, s.question, tuple(s.gold_answers)) for s in load_samples(cfg_c)]
    assert a == b
    assert a != c
    assert len(a) == 4
    for sid, q, golds in a:
        assert sid.startswith("fake:validation:")
        assert "number" in q.lower()
        assert len(golds) == 1
