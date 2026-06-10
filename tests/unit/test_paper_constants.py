"""Per-model paper constants: App E.1 layer ranges + App G.1 max_alpha."""

from __future__ import annotations

import pytest

from sae_muc.pipeline.intervene import _resolve_alpha_max
from sae_muc.pipeline.paper_layer_ranges import paper_layer_range, paper_max_alpha
from sae_muc.pipeline.probe_layer_ranges import probe_layer_range, probe_layer_union


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("meta-llama/Llama-3.1-8B-Instruct", 1.0),
        ("mistralai/Mistral-7B-Instruct-v0.3", 0.4),
        ("Qwen/Qwen2.5-7B-Instruct", 3.0),
        ("meta-llama/Llama-3.1-70B-Instruct", 4.0),  # 70B beats generic llama
    ],
)
def test_paper_max_alpha(model_name, expected):
    assert paper_max_alpha(model_name) == expected


def test_paper_max_alpha_unknown_model_raises():
    with pytest.raises(ValueError, match="App G.1"):
        paper_max_alpha("google/gemma-2-2b")


def test_resolve_alpha_max_paper_keyword():
    assert _resolve_alpha_max("paper", "Qwen/Qwen2.5-7B-Instruct") == 3.0


def test_resolve_alpha_max_explicit_float():
    assert _resolve_alpha_max(0.5, "any-model") == 0.5


def test_paper_layer_range_qwen_matches_app_e1():
    assert paper_layer_range("Qwen/Qwen2.5-7B-Instruct") == list(range(16, 28))


def test_alpha_max_paper_validates_in_config():
    from sae_muc.config import InterveneStage

    assert InterveneStage(alpha_max="paper").alpha_max == "paper"
    assert InterveneStage(alpha_max=0.4).alpha_max == 0.4


@pytest.mark.parametrize(
    ("dataset", "uncertainty", "expected"),
    [
        ("triviaqa", "vu", list(range(5, 21))),
        ("nq_open", "vu", list(range(10, 21))),
        ("popqa", "vu", list(range(5, 21))),
        ("triviaqa", "su", list(range(10, 21))),
        ("nq_open", "su", list(range(10, 21))),
        ("popqa", "su", list(range(5, 26))),
    ],
)
def test_probe_layer_range_app_f1(dataset, uncertainty, expected):
    assert probe_layer_range(dataset, uncertainty) == expected


def test_probe_layer_union_popqa():
    # VU 5-20 ∪ SU 5-25 = 5-25.
    assert probe_layer_union("popqa") == list(range(5, 26))


def test_probe_layer_range_unknown_raises():
    with pytest.raises(ValueError, match="App F.1"):
        probe_layer_range("squad", "vu")
