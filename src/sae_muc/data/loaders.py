"""Load TriviaQA / NQ-Open / PopQA from HuggingFace, normalise to `Sample`.

The HF `datasets` call sits behind `_hf_load_dataset` so tests can swap in a
fake without touching the network or the heavy `datasets` import.
"""

from __future__ import annotations

import json
from typing import Any

from sae_muc.config import DatasetConfig
from sae_muc.data.schema import Sample


def _hf_load_dataset(repo: str, *args: Any, **kwargs: Any) -> Any:
    """Indirection layer around `datasets.load_dataset` to keep tests fast."""
    from datasets import load_dataset

    return load_dataset(repo, *args, **kwargs)


def load_samples(cfg: DatasetConfig) -> list[Sample]:
    """Load, shuffle deterministically, take the first `n_samples`, normalise."""
    if cfg.name == "triviaqa":
        return _load_triviaqa(cfg)
    if cfg.name == "nq_open":
        return _load_nq_open(cfg)
    if cfg.name == "popqa":
        return _load_popqa(cfg)
    if cfg.name == "fake":
        return _load_fake(cfg)
    raise ValueError(f"Unknown dataset: {cfg.name!r}")


def _take(ds: Any, cfg: DatasetConfig) -> Any:
    """Shuffle with the configured seed and take up to `n_samples`."""
    ds = ds.shuffle(seed=cfg.seed)
    n = min(cfg.n_samples, len(ds))
    return ds.select(range(n))


def _load_triviaqa(cfg: DatasetConfig) -> list[Sample]:
    ds = _take(_hf_load_dataset("mandarjoshi/trivia_qa", "rc", split=cfg.split), cfg)
    samples: list[Sample] = []
    for i, row in enumerate(ds):
        answer = row.get("answer") or {}
        value = answer.get("value") or ""
        aliases = list(answer.get("aliases") or [])
        # Order: primary value first, then aliases; drop empties and duplicates.
        gold = [value, *aliases] if value else aliases
        gold = list(dict.fromkeys(g for g in gold if g))
        samples.append(
            Sample(
                sample_id=f"triviaqa:{cfg.split}:{i}",
                question=row["question"],
                gold_answers=gold,
            )
        )
    return samples


def _load_nq_open(cfg: DatasetConfig) -> list[Sample]:
    ds = _take(_hf_load_dataset("google-research-datasets/nq_open", split=cfg.split), cfg)
    samples: list[Sample] = []
    for i, row in enumerate(ds):
        samples.append(
            Sample(
                sample_id=f"nq_open:{cfg.split}:{i}",
                question=row["question"],
                gold_answers=list(row["answer"]),
            )
        )
    return samples


def _load_fake(cfg: DatasetConfig) -> list[Sample]:
    """Synthetic in-memory dataset for offline smoke tests.

    No network, no HF download. Deterministic from `cfg.seed` and
    `cfg.n_samples`. Each question asks for an integer and the golden
    answer is the same number written out — good enough to exercise the
    full pipeline on fakes without any external state.
    """
    _WORDS = [
        "zero", "one", "two", "three", "four",
        "five", "six", "seven", "eight", "nine",
    ]
    samples: list[Sample] = []
    for i in range(cfg.n_samples):
        num = (cfg.seed + i) % len(_WORDS)
        samples.append(
            Sample(
                sample_id=f"fake:{cfg.split}:{i}",
                question=f"What is the word for the number {num}?",
                gold_answers=[_WORDS[num]],
            )
        )
    return samples


def _load_popqa(cfg: DatasetConfig) -> list[Sample]:
    ds = _take(_hf_load_dataset("akariasai/PopQA", split=cfg.split), cfg)
    samples: list[Sample] = []
    for i, row in enumerate(ds):
        possible = row.get("possible_answers")
        if isinstance(possible, str):
            gold = json.loads(possible)
        elif isinstance(possible, list):
            gold = list(possible)
        else:
            # Fallback: PopQA rows always have `obj` (the object entity name).
            obj = row.get("obj")
            gold = [obj] if obj else []
        samples.append(
            Sample(
                sample_id=f"popqa:{cfg.split}:{i}",
                question=row["question"],
                gold_answers=[g for g in gold if g],
            )
        )
    return samples
