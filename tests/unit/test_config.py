from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from sae_muc.config import (
    ExperimentConfig,
    _slug,
    load_experiment_config,
    load_yaml_with_extends,
)


def _write(path: Path, body: str) -> Path:
    path.write_text(textwrap.dedent(body).lstrip())
    return path


def test_slug_normalises_model_names():
    assert _slug("mistralai/Mistral-7B-Instruct-v0.3") == "mistralai_mistral_7b_instruct_v0_3"
    assert _slug("nq_open") == "nq_open"


def test_minimal_config_validates():
    cfg = ExperimentConfig.model_validate(
        {
            "model": {"provider": "fake", "name": "fake-7b"},
            "dataset": {"name": "nq_open"},
            "judge": {"provider": "fake", "model": "fake-70b"},
        }
    )
    assert cfg.seed == 42
    assert cfg.stages.generate.n_samples == 10
    assert cfg.model.dtype == "bfloat16"


def test_extra_fields_rejected():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "model": {"provider": "fake", "name": "x", "unknown_field": 1},
                "dataset": {"name": "nq_open"},
                "judge": {"provider": "fake", "model": "y"},
            }
        )


def test_config_hash_is_stable_and_order_invariant():
    cfg_a = ExperimentConfig.model_validate(
        {
            "model": {"provider": "fake", "name": "a"},
            "dataset": {"name": "nq_open"},
            "judge": {"provider": "fake", "model": "b"},
        }
    )
    cfg_b = ExperimentConfig.model_validate(
        {
            "judge": {"model": "b", "provider": "fake"},
            "dataset": {"name": "nq_open"},
            "model": {"name": "a", "provider": "fake"},
        }
    )
    assert cfg_a.config_hash() == cfg_b.config_hash()


def test_build_run_id_format(tmp_path):
    cfg = ExperimentConfig.model_validate(
        {
            "model": {"provider": "fake", "name": "Mistral-7B"},
            "dataset": {"name": "nq_open"},
            "judge": {"provider": "fake", "model": "J"},
        }
    )
    rid = cfg.build_run_id("20260423-150000")
    parts = rid.split("_")
    assert parts[0] == "nq"  # after slugification, the underscore ends a segment
    # hash8 is the trailing segment of length 8
    assert len(parts[-1]) == 8
    assert "20260423" in rid


def test_extends_merges_nested(tmp_path):
    parent = _write(
        tmp_path / "base.yaml",
        """
        seed: 1
        stages:
          generate:
            n_samples: 10
            max_new_tokens: 100
        """,
    )
    child = _write(
        tmp_path / "child.yaml",
        f"""
        extends: {parent.name}
        stages:
          generate:
            n_samples: 3
        """,
    )
    merged = load_yaml_with_extends(child)
    assert merged["seed"] == 1
    assert merged["stages"]["generate"]["n_samples"] == 3
    assert merged["stages"]["generate"]["max_new_tokens"] == 100


def test_load_experiment_config_end_to_end(tmp_path):
    _write(
        tmp_path / "base.yaml",
        """
        seed: 7
        stages:
          generate:
            n_samples: 5
        """,
    )
    cfg_path = _write(
        tmp_path / "exp.yaml",
        """
        extends: base.yaml
        model: {provider: fake, name: fake-7b}
        dataset: {name: nq_open, n_samples: 20}
        judge: {provider: fake, model: fake-70b}
        """,
    )
    cfg = load_experiment_config(cfg_path)
    assert cfg.seed == 7
    assert cfg.dataset.n_samples == 20
    assert cfg.stages.generate.n_samples == 5
