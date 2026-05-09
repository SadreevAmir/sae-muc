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
    rid = cfg.build_run_id("20260423-150000", user="k.frolov")
    parts = rid.split("__")
    assert len(parts) == 5  # user, dataset, model, ts, hash8
    assert parts[0] == "k_frolov"
    assert parts[1] == "nq_open"
    assert parts[2] == "mistral_7b"
    assert parts[3] == "20260423-150000"
    assert len(parts[4]) == 8


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


def test_extends_accepts_list(tmp_path):
    _write(
        tmp_path / "a.yaml",
        """
        seed: 1
        stages:
          generate: {n_samples: 10}
        """,
    )
    _write(
        tmp_path / "b.yaml",
        """
        seed: 2
        stages:
          generate: {temperature_high: 0.9}
        """,
    )
    cfg_path = _write(
        tmp_path / "exp.yaml",
        """
        extends: [a.yaml, b.yaml]
        model: {provider: fake, name: x}
        dataset: {name: nq_open}
        judge: {provider: fake, model: j}
        """,
    )
    cfg = load_experiment_config(cfg_path)
    # Later entries override earlier ones; exp yaml itself overrides both.
    assert cfg.seed == 2
    assert cfg.stages.generate.n_samples == 10
    assert cfg.stages.generate.temperature_high == 0.9


def test_string_section_ref_resolves_to_file(tmp_path):
    _write(
        tmp_path / "model_fake.yaml",
        """
        provider: fake
        name: referenced-model
        """,
    )
    _write(
        tmp_path / "dataset_fake.yaml",
        """
        name: nq_open
        split: validation
        n_samples: 42
        """,
    )
    _write(
        tmp_path / "judge_fake.yaml",
        """
        provider: fake
        model: referenced-judge
        """,
    )
    _write(
        tmp_path / "nli_fake.yaml",
        """
        provider: fake
        model: referenced-nli
        """,
    )
    cfg_path = _write(
        tmp_path / "exp.yaml",
        """
        model: model_fake.yaml
        dataset: dataset_fake.yaml
        judge: judge_fake.yaml
        nli: nli_fake.yaml
        """,
    )
    cfg = load_experiment_config(cfg_path)
    assert cfg.model.name == "referenced-model"
    assert cfg.dataset.n_samples == 42
    assert cfg.judge.model == "referenced-judge"
    assert cfg.nli.model == "referenced-nli"


def test_inline_dict_still_works_alongside_refs(tmp_path):
    _write(
        tmp_path / "model_fake.yaml",
        """
        provider: fake
        name: from-file
        """,
    )
    cfg_path = _write(
        tmp_path / "exp.yaml",
        """
        model: model_fake.yaml
        dataset: {name: nq_open, n_samples: 5}
        judge: {provider: fake, model: inline}
        """,
    )
    cfg = load_experiment_config(cfg_path)
    assert cfg.model.name == "from-file"
    assert cfg.dataset.n_samples == 5
    assert cfg.judge.model == "inline"


def test_shipped_smoke_configs_validate():
    """The YAMLs we ship in configs/experiment/ must all parse."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    for path in sorted((repo_root / "configs/experiment").glob("*.yaml")):
        load_experiment_config(path)
