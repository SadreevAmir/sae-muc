from __future__ import annotations

import pandas as pd
import pytest

from sae_muc.artifacts import ArtifactStore, make_run
from sae_muc.config import ExperimentConfig


def _cfg(tmp_path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "model": {"provider": "fake", "name": "fake-7b"},
            "dataset": {"name": "nq_open"},
            "judge": {"provider": "fake", "model": "j"},
            "data_root": str(tmp_path / "data"),
        }
    )


def test_make_run_creates_dir_and_freezes_config(tmp_path):
    cfg = _cfg(tmp_path)
    run_id, store = make_run(cfg)
    assert (cfg.data_root / "runs" / run_id).is_dir()
    assert store.exists("config.resolved.json")
    saved = store.load_json("config.resolved.json")
    assert saved["model"]["name"] == "fake-7b"


def test_make_run_reuse_run_id(tmp_path):
    cfg = _cfg(tmp_path)
    run_id, store = make_run(cfg)
    # Re-entering with the same run_id does not rewrite the frozen config
    frozen_before = store.load_json("config.resolved.json")
    run_id2, store2 = make_run(cfg, run_id=run_id)
    assert run_id2 == run_id
    assert store2.load_json("config.resolved.json") == frozen_before


def test_parquet_roundtrip(tmp_path):
    store = ArtifactStore(tmp_path / "run")
    df = pd.DataFrame({"sample_id": ["a", "b"], "value": [1.0, 2.0]})
    store.save_parquet("out.parquet", df)
    assert store.exists("out.parquet")
    pd.testing.assert_frame_equal(store.load_parquet("out.parquet"), df)


def test_json_roundtrip(tmp_path):
    store = ArtifactStore(tmp_path / "run")
    payload = {"k": 1, "list": [1, 2, 3]}
    store.save_json("r.json", payload)
    assert store.load_json("r.json") == payload


def test_safetensors_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    store = ArtifactStore(tmp_path / "run")
    tensors = {"a": torch.arange(6).reshape(2, 3).contiguous()}
    store.save_safetensors("nested/tensors.safetensors", tensors)
    loaded = store.load_safetensors("nested/tensors.safetensors")
    assert torch.equal(loaded["a"], tensors["a"])
