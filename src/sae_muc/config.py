"""Config schema and loader for sae-muc experiments.

Configs are plain YAML with optional one-level `extends:` merging. Loaded
configs are validated into frozen pydantic models; the resolved config is
written alongside run artefacts for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelConfig(_Frozen):
    provider: Literal["hf_local", "openrouter", "fake"]
    name: str
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"


class DatasetConfig(_Frozen):
    name: Literal["triviaqa", "nq_open", "popqa", "fake"]
    split: str = "validation"
    n_samples: int = 500
    seed: int = 42


class JudgeConfig(_Frozen):
    provider: Literal["openrouter", "cherryin", "fake"]
    model: str
    max_retries: int = 3


class NLIConfig(_Frozen):
    provider: Literal["hf_local", "fake"] = "hf_local"
    model: str = "microsoft/deberta-v2-xxlarge-mnli"


class GenerateStage(_Frozen):
    n_samples: int = 10
    temperature_low: float = 0.1
    temperature_high: float = 1.0
    max_new_tokens: int = 100


class HiddenStatesStage(_Frozen):
    # `full` — forward `question + greedy_answer`, store every token.
    # `question_only` — forward only the question text (no answer).
    #     This is enough for paper-faithful pooling=last_token_q / mean_q
    #     and saves 2-3× memory/time vs full.
    # `last_k_tokens` — forward full text but store only the last `last_k`
    #     tokens. Useful when the question is long and only the tail matters
    #     for the downstream probe. NB: if the question itself doesn't fit
    #     into the kept tail, pooling=last_token_q will fall back to the
    #     last *stored* token, which may belong to the answer.
    storage: Literal["full", "question_only", "last_k_tokens"] = "full"
    last_k: int = 8


class VUFStage(_Frozen):
    layers: list[int] | Literal["auto"] = "auto"
    # selection chooses how the (uncertain, certain) splits are drawn:
    #   "top_n"       — top n_top uncertain + bottom n_bot certain by VU rank
    #                   (paper §3.1 protocol for VUF *validation* / Fig.4).
    #   "vu_threshold" — VU ≥ vu_uncertain_min for uncertain,
    #                    VU ≤ vu_certain_max  for certain
    #                   (paper App G.1 protocol for *mitigation*).
    # Default is "vu_threshold" because the mitigation pipeline (intervene,
    # sae_features) is what `vuf/splits.parquet` mostly feeds.
    selection: Literal["top_n", "vu_threshold"] = "vu_threshold"
    n_top: int = 250
    n_bot: int = 250
    vu_uncertain_min: float = 0.9
    vu_certain_max: float = 0.05
    pooling: Literal["last_token_q", "last_token_a", "mean_q", "mean_a"] = "last_token_q"


class InterveneStage(_Frozen):
    method: Literal["linear_vuf", "sae_emd", "sae_projected", "sae_clamp"] = "linear_vuf"
    # fixed: iterate alpha_grid, same α for every question (paper Fig.5/6 ablation).
    # adaptive: per-question α_su(x) = clip(SU_norm(x) − VU(x), 0, alpha_max),
    #           paper §4.2 Eq.5–6; this is the MUC intervention proper.
    mode: Literal["fixed", "adaptive"] = "fixed"
    alpha_grid: list[float] = Field(default_factory=lambda: [-1.0, -0.5, 0.0, 0.5, 1.0])
    alpha_max: float = 1.0
    # Layer(s) where the hook is registered. The paper applies the linear VUF
    # to a contiguous range of layers (App E.1, Tab.5):
    #   "paper_range" — per-model App E.1 range (Llama 15-31, Mistral 15-31,
    #                   Qwen 16-27); see pipeline/paper_layer_ranges.py.
    #   list[int]     — explicit layer indices.
    #   int           — single layer.
    #   "auto"        — middle of the available VUF layers (single layer).
    layer: int | list[int] | Literal["auto", "paper_range"] = "auto"
    # sae_clamp only: target activation value for selected uncertainty
    # features (certainty features are clamped to 0). α multiplies this.
    sae_clamp_target: float = 10.0


class DetectStage(_Frozen):
    # Refusal is classified from the judge's VU on the greedy answer
    # (paper §3.2: "we categorize the samples ... based on the VU level
    # of the most likely answer"). A greedy answer with VU ≥ threshold is
    # considered a refusal/abstention and excluded from hallucination
    # training.
    #
    # Paper note: §3.2 introduces the categorisation but does not pin down
    # a specific cut-off value. 0.85 is OUR calibration default — Tab.3-style
    # numbers under this threshold are not directly comparable to the paper.
    # Override per experiment via YAML (`stages.detect.refusal_vu_threshold`).
    refusal_vu_threshold: float = 0.85


class SAEFeaturesStage(_Frozen):
    # Top K features to mark as 'uncertainty' (positive Cohen's d) and K to
    # mark as 'certainty' (negative d). Paper prototype used ~50.
    k_top: int = 50


class EvaluateStage(_Frozen):
    """Thresholds for Tab.3 metrics.

    `kossen` mode (paper-faithful for the "Confident Hallucination Rate"
    bullet): the threshold minimises `sum((value - t)**2)` over the baseline
    distribution of values, which collapses analytically to the mean.
    `fixed` mode uses the explicit `vu_threshold` / `su_threshold` floats.
    """
    vu_threshold_mode: Literal["fixed", "kossen"] = "kossen"
    vu_threshold: float = 0.5
    su_threshold_mode: Literal["fixed", "kossen", "median"] = "kossen"
    su_threshold: float | None = None


class StagesConfig(_Frozen):
    generate: GenerateStage = GenerateStage()
    hidden_states: HiddenStatesStage = HiddenStatesStage()
    vuf: VUFStage = VUFStage()
    intervene: InterveneStage = InterveneStage()
    detect: DetectStage = DetectStage()
    sae_features: SAEFeaturesStage = SAEFeaturesStage()
    evaluate: EvaluateStage = EvaluateStage()


class SAEConfig(_Frozen):
    """Where the SAE comes from, plus dimensions for the fake backend.

    provider=fake uses an in-process random linear projection for tests.
    provider=sae_lens loads a pretrained SAE by (release, sae_id). In that
    case d_in/d_latent are ignored — they come from the SAE config. For
    the fake backend, d_in must match the hidden-state dimensionality of
    the generator model (e.g. 896 for Qwen2.5-0.5B; 8 for FakeBackend).
    """

    provider: Literal["fake", "sae_lens"] = "fake"
    d_in: int = 8
    d_latent: int = 16
    seed: int = 42
    release: str | None = None
    sae_id: str | None = None


class ExperimentConfig(_Frozen):
    model: ModelConfig
    dataset: DatasetConfig
    judge: JudgeConfig
    nli: NLIConfig = NLIConfig()
    sae: SAEConfig = SAEConfig()
    stages: StagesConfig = StagesConfig()
    seed: int = 42
    data_root: Path = Path("data")

    def config_hash(self) -> str:
        """Stable SHA-256 hex digest of the resolved config (full length)."""
        payload = self.model_dump(mode="json")
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def build_run_id(self, timestamp: str) -> str:
        """Build a human-readable run_id: `<dataset>_<model>_<timestamp>_<hash8>`."""
        return f"{_slug(self.dataset.name)}_{_slug(self.model.name)}_{timestamp}_{self.config_hash()[:8]}"


def _slug(s: str) -> str:
    """Lower-case; replace any non-alphanumeric run with a single underscore."""
    out: list[str] = []
    prev_under = False
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
            prev_under = False
        elif not prev_under:
            out.append("_")
            prev_under = True
    return "".join(out).strip("_")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# Fields that can be set to a string path pointing at a standalone YAML
# fragment (e.g. `model: ../model/mistral7b.yaml`). The loader replaces the
# string with the file's content before validation. Inline dicts and/or
# extends-based merging are unaffected.
_REFERENCE_SECTIONS: tuple[str, ...] = ("model", "dataset", "judge", "nli")


def load_yaml_with_extends(path: Path) -> dict[str, Any]:
    """Load a YAML file; resolve `extends:` and string-valued section refs.

    `extends:` may be a single relative path (str) or a list of paths,
    resolved in order with later entries overriding earlier ones. The
    current file's keys override the merged `extends:` result.

    After the extends merge, any field in `_REFERENCE_SECTIONS` whose value
    is still a string is interpreted as a relative path to a YAML fragment
    and loaded in place. Inline dict values pass through unchanged.
    """
    path = Path(path).resolve()
    with path.open() as f:
        raw = yaml.safe_load(f) or {}

    extends = raw.pop("extends", None)
    if extends is not None:
        if isinstance(extends, str):
            extends_list = [extends]
        elif isinstance(extends, list):
            extends_list = list(extends)
        else:
            raise TypeError(
                f"`extends:` must be a string or list of strings, "
                f"got {type(extends).__name__}"
            )
        merged: dict[str, Any] = {}
        for ext in extends_list:
            if not isinstance(ext, str):
                raise TypeError(f"`extends:` entries must be strings, got {ext!r}")
            parent_path = (path.parent / ext).resolve()
            parent = load_yaml_with_extends(parent_path)
            merged = _deep_merge(merged, parent)
        raw = _deep_merge(merged, raw)

    for section in _REFERENCE_SECTIONS:
        val = raw.get(section)
        if isinstance(val, str):
            ref_path = (path.parent / val).resolve()
            with ref_path.open() as f:
                raw[section] = yaml.safe_load(f) or {}

    return raw


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML config file, resolve `extends:`/refs, validate into ExperimentConfig."""
    raw = load_yaml_with_extends(Path(path))
    return ExperimentConfig.model_validate(raw)
