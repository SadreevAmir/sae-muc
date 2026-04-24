"""Shared fixtures: synthetic mini-dataset + FakeBackend-based pipeline context."""

from __future__ import annotations

import pytest

from sae_muc.config import ExperimentConfig
from sae_muc.pipeline.context import PipelineContext, build_context


@pytest.fixture
def fake_hf_rows(monkeypatch):
    """Patch HF `load_dataset` to return a small in-memory NQ-Open-shaped dataset."""

    class _FakeDS:
        def __init__(self, rows):
            self.rows = list(rows)

        def shuffle(self, seed):
            # Deterministic fake shuffle: identity (order stays as given).
            return _FakeDS(self.rows)

        def select(self, idx):
            return _FakeDS([self.rows[i] for i in idx])

        def __len__(self):
            return len(self.rows)

        def __iter__(self):
            return iter(self.rows)

    rows = [
        {"question": "What is the capital of France?", "answer": ["Paris"]},
        {"question": "Who wrote War and Peace?", "answer": ["Leo Tolstoy", "Tolstoy"]},
        {"question": "What year did WWII end?", "answer": ["1945"]},
        {"question": "Who painted the Mona Lisa?", "answer": ["Leonardo da Vinci"]},
        {"question": "What is the largest ocean?", "answer": ["Pacific Ocean"]},
    ]

    def fake_load(repo, *args, **kwargs):
        assert repo == "google-research-datasets/nq_open"
        return _FakeDS(rows)

    monkeypatch.setattr("sae_muc.data.loaders._hf_load_dataset", fake_load)
    return rows


@pytest.fixture
def fake_cfg(tmp_path) -> ExperimentConfig:
    """A minimal all-fakes experiment config pinned to `tmp_path / data`."""
    return ExperimentConfig.model_validate(
        {
            "model": {"provider": "fake", "name": "fake-7b"},
            "dataset": {"name": "nq_open", "split": "validation", "n_samples": 5},
            "judge": {"provider": "fake", "model": "fake-judge"},
            "nli": {"provider": "fake", "model": "fake-nli"},
            # FakeBackend._D_MODEL = 8; SAE d_in must match.
            "sae": {"provider": "fake", "d_in": 8, "d_latent": 16, "seed": 7},
            "data_root": str(tmp_path / "data"),
            "stages": {
                "generate": {"n_samples": 3, "max_new_tokens": 16},
            },
        }
    )


@pytest.fixture
def fake_ctx(fake_cfg, fake_hf_rows) -> PipelineContext:
    _rid, ctx = build_context(fake_cfg)
    return ctx
