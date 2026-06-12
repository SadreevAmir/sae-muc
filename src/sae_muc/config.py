"""Config schema and loader for sae-muc experiments.

Configs are plain YAML with optional one-level `extends:` merging. Loaded
configs are validated into frozen pydantic models; the resolved config is
written alongside run artefacts for reproducibility.
"""

from __future__ import annotations

import getpass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    # Number of judge calls kept in flight at once via a continuous worker
    # pool (judge / accuracy_judge stages). 1 (default) is sequential and
    # byte-identical to the pre-pool behaviour; raise it for remote backends
    # that tolerate concurrency (CherryIn load-tested fine at 8).
    concurrency: int = 1

    @field_validator("concurrency")
    @classmethod
    def _check_concurrency(cls, v: int) -> int:
        if v < 1:
            raise ValueError("judge.concurrency must be >= 1")
        return v


class NLIConfig(_Frozen):
    provider: Literal["hf_local", "fake"] = "hf_local"
    model: str = "microsoft/deberta-v2-xxlarge-mnli"


class GenerateStage(_Frozen):
    n_samples: int = 10
    temperature_low: float = 0.1
    temperature_high: float = 1.0
    max_new_tokens: int = 100
    # Which Appendix-A.1 prompt(s) to generate answers with:
    #   "split" (default, paper-faithful) — generate BOTH a plain set
    #       (App A.1 box 1 → Semantic Entropy + accuracy + most-likely) and
    #       an eliciting/hedging set (App A.1 box 2 → VU / VUF, §3.1).
    #       App C feeds the *plain* question to the SE samples; the hedging
    #       prompt is bound to VU only. Doubles generation cost.
    #   "eliciting_only" — generate only the eliciting set and reuse it for
    #       everything (cheaper; the pre-split behaviour, for smokes).
    prompt_regime: Literal["split", "eliciting_only"] = "split"
    # Prompt-batch size for the HF-local backend. The effective GPU batch at
    # generation is `batch_size * n` (num_return_sequences), so this caps peak
    # activation/KV-cache memory independently of N. 0 (default) means "all
    # prompts in one forward" — byte-identical to the pre-batching path, so
    # smokes stay unchanged. Set it on large-N runs (e.g. 256 → ≤256*10=2560
    # sequences per sampled forward on a 48GB card). Ignored by remote/fake
    # backends.
    batch_size: int = 0
    # Nucleus / top-K truncation for sampled decoding (paper App C p.18:
    # "temperature of 1 with nucleus sampling (P = 0.9) and top-K sampling
    # (K = 50)"). Pinned here so the high-T sample distribution feeding SE
    # clustering and the per-question VU mean is model-independent; without
    # them HF falls back to each model's bundled generation_config. Only
    # applied when do_sample (temperature > 0); set to None to defer to the
    # backend/model default.
    top_p: float | None = 0.9
    top_k: int | None = 50


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
    # CPU dtype of the saved residual-stream tensors. `float32` (default) keeps
    # existing artefacts byte-identical. `bfloat16` halves host-RAM peak during
    # extraction and the on-disk safetensors size at no downstream cost
    # (pooling / Cohen's d / Ridge all run fine in bf16) — set it on large-N
    # runs. The HF-local backend casts to this dtype before moving to CPU;
    # the fake backend keeps float32 (tests assert exact values).
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"


class CategorizeStage(_Frozen):
    """LLM-as-judge categorisation of uncertain-bucket generations.

    Splits the `uncertain = {VU ≥ vu_uncertain_min}` set into ABSTAIN
    ("I don't know") vs HEDGE ("I think X") so the downstream `vuf` stage
    can build per-category directions and the disentanglement question
    (paper Tab.3 mixing of refusal + hedging) has a scalar answer.

    Opt-in by design: `enabled=False` (default) keeps every existing run
    byte-identical. Reuses `ctx.judge` — no second backend.
    """
    enabled: bool = False
    max_retries: int = 3


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
    # Universal / OOD VUF (paper App G.1 / Table 5). When non-empty, the
    # `combine_vuf` stage pools the per-set means saved by these source runs
    # into one diff-in-means VUF per layer and overwrites this run's
    # vuf/direction_layer_*.safetensors:
    #   * three datasets (TriviaQA+NQ-Open+PopQA) → the universal VUF;
    #   * a single source → OOD reuse (e.g. TriviaQA VUF applied here, Table 5).
    # Each entry is a run directory path or a run_id under data_root/runs/.
    # Empty (default) → combine_vuf is a no-op and the single-dataset VUF stands.
    combine_sources: list[str] = Field(default_factory=list)
    # When True AND `categories.parquet` has labelled question-level rows,
    # `vuf.run` also builds per-category directions (`abstain`, `hedge`)
    # in addition to the aggregated `main` one — supervisor's
    # disentanglement experiment. Off by default for byte-identical
    # back-compat with existing YAMLs.
    per_category: bool = False


class InterveneStage(_Frozen):
    method: Literal["linear_vuf", "sae_emd", "sae_projected", "sae_clamp"] = "linear_vuf"
    # fixed: iterate alpha_grid, same α for every question (paper Fig.5/6 ablation).
    # adaptive: per-question α_su(x) = clip((SU_norm(x) − VU(x))·alpha_max, 0, alpha_max),
    #           paper §4.2 Eq.5–6; this is the MUC intervention proper.
    mode: Literal["fixed", "adaptive"] = "fixed"
    alpha_grid: list[float] = Field(default_factory=lambda: [-1.0, -0.5, 0.0, 0.5, 1.0])
    # Adaptive (MUC) steering gain for Eq.6 clip((su_norm − vu)·alpha_max, 0, alpha_max).
    #   float    — explicit gain (default 1.0).
    #   "paper"  — per-model App G.1 value via paper_max_alpha(): Llama-3.1-8B
    #              1.0, Mistral-7B 0.4, Qwen2.5-7B 3.0, Llama-3.1-70B 4.0.
    # The 1.0 default matches Llama-8B by coincidence; Qwen/Mistral runs must
    # set "paper" (or the right float) — a global 1.0 under-steers Qwen 3×.
    #   list   — run each ceiling as its own adaptive variant in ONE run,
    #            reusing all upstream artefacts. Output lands in per-ceiling
    #            dirs intervention/adaptive_amax_{v:+.2f}; meta.parquet gets one
    #            row per ceiling. A scalar (float/"paper") keeps the legacy
    #            single intervention/adaptive dir + single meta row.
    alpha_max: float | Literal["paper"] | list[float | Literal["paper"]] = 1.0
    # Layer(s) where the hook is registered. The paper applies the linear VUF
    # to a contiguous range of layers (App E.1, Tab.5):
    #   "paper_range" — per-model App E.1 range (Llama 15-31, Mistral 15-31,
    #                   Qwen 16-27); see pipeline/paper_layer_ranges.py.
    #   list[int]     — explicit layer indices.
    #   int           — single layer.
    #   "auto"        — middle of the available VUF layers (single layer).
    layer: int | list[int] | Literal["auto", "paper_range"] = "auto"
    # sae_emd only: how the latent shift δ is built.
    #   "cohen_d"  — δ[i] = cohen_d[i] for selected features, then L2-normalised
    #                (paper-faithful per old prototype build_intervention_config_v2).
    #   "multihot" — δ[i] = +1 for uncertainty, -1 for certainty (uniform shift).
    sae_emd_delta: Literal["cohen_d", "multihot"] = "cohen_d"
    # When False, the hook only fires during prefill (seq_len > 1) and is
    # skipped on each autoregressive decode step (seq_len == 1). Matches
    # old_prototype hooks.py:apply_during_generation. Paper §4.2 says
    # "all tokens in detected hallucinated responses", so True is the
    # paper-faithful default; False is for back-compat with old experiments.
    apply_during_generation: bool = True
    # Detector gating (paper §4.2: "we modulate the influence of these
    # features through linear interventions on all tokens in detected
    # hallucinated responses"). When True, the adaptive run reads
    # detection.parquet and skips the intervention for samples whose
    # hallucination probability is below `detector_threshold` — those
    # rows are copied verbatim from the baseline generations.parquet
    # with alpha=0. fixed mode ignores the gate (with a warning) since
    # the alpha-grid sweep is meant to perturb all questions uniformly.
    gate_by_detector: bool = False
    detector_threshold: float = 0.5
    # Which detection.parquet column to read for the gate. Defaults to the
    # column produced by the active stages.detect.detector_method:
    #   "auto" — pick combined / hidden / combined_full per detector_method.
    #   else  — read prob_hallucinate_<gate_detector_method> directly.
    gate_detector_method: Literal["auto", "verbal", "semantic", "combined", "hidden", "combined_full"] = "auto"

    @field_validator("alpha_max")
    @classmethod
    def _check_alpha_max(cls, v):
        """A list ceiling-sweep must be non-empty; each entry is "paper" or > 0.

        The SCALAR path is intentionally left unvalidated for back-compat (a
        non-positive scalar was always accepted); only the new list feature
        enforces the entries-must-be-positive guarantee.
        """
        if not isinstance(v, list):
            return v
        if not v:
            raise ValueError("intervene.alpha_max list must not be empty")
        for entry in v:
            if entry == "paper":
                continue
            if not (isinstance(entry, (int, float)) and entry > 0):
                raise ValueError(
                    f"intervene.alpha_max entries must be \"paper\" or a number "
                    f"> 0, got {entry!r}"
                )
        return v


class DetectStage(_Frozen):
    # Refusal is classified from the judge's VU on the greedy answer
    # (paper §2.3 p.4: "we categorize the samples ... based on the VU level
    # of the most likely answer"). A greedy answer with VU ≥ threshold is
    # considered a refusal/abstention and excluded from hallucination
    # training.
    #
    # Paper note: §2.3 defines abstention BEHAVIOURALLY (an abstained answer
    # punts → decisiveness 0 → VU 1) and pins NO numeric VU cut-off. 0.85 is
    # OUR calibration default — Tab.3-style numbers under it are not directly
    # comparable to the paper. The behavioural consistently/partly-abstained
    # split (Table 6) is reproduced as a diagnostic in evaluate.py.
    # Override per experiment via YAML (`stages.detect.refusal_vu_threshold`).
    refusal_vu_threshold: float = 0.85
    # Hallucination-detector model (paper §4.1):
    #   "lr_vu_se"  — three LRs over (vu), (se), (vu, se). Default; cheapest.
    #   "lr_hidden" — also fits an LR on the pooled hidden state at
    #                 detector_layer (paper Tab.1 reports this is best).
    #   "combined"  — additionally fits an LR on (vu, se, hidden) concat.
    # In all modes the verbal / semantic / combined LRs are still trained
    # so per-feature comparisons stay available in detection_metrics.json.
    detector_method: Literal["lr_vu_se", "lr_hidden", "combined"] = "lr_vu_se"
    # Detector input features (paper §4.1 / Table 2):
    #   "calculated"      — feed the LR the calculated VU (judge mean) and SU
    #                       (semantic entropy). Requires full sampling.
    #   "probe_predicted" — feed the LR the VU/SU PREDICTED by two linear
    #                       regressor probes over the question's last-token
    #                       hidden state (App F.1 ranges); no sampling needed at
    #                       inference. Reproduces Table 2's Probe-Predicted column.
    detector_input: Literal["calculated", "probe_predicted"] = "calculated"
    # Layer(s) for the hidden-state classifier probe (lr_hidden / combined):
    #   int / list[int] — explicit; "auto" — middle of the available VUF layers
    #   (single layer); "paper_range" — App F.1 per-dataset union of the VU+SU
    #   probe ranges (probe_layer_ranges.py). Independent of intervene.layer.
    detector_layer: int | list[int] | Literal["auto", "paper_range"] = "auto"
    # External Table-2 baselines to additionally fit (paper §4.1 Baselines):
    #   "sep" — Semantic Entropy Probe: LR over the TBG (last-question-token)
    #           hidden state predicting binarized SU, abstained→non-hallucinated.
    # Each needs hidden states; empty by default (no extra cost).
    detector_baselines: list[Literal["sep"]] = Field(default_factory=list)


class SAEFeaturesStage(_Frozen):
    # Top K features to mark as 'uncertainty' (positive Cohen's d) and K to
    # mark as 'certainty' (negative d). Paper prototype used ~50.
    k_top: int = 50
    # Selection algorithm:
    #   "topk"      — take the |d|-largest k_top features in each direction
    #                 (cheap, deterministic).
    #   "consensus" — old prototype's robust filter
    #     (build_intervention_config_v2.py:207-226):
    #     bootstrap-stable AND (BH-FDR significant OR |d| > cohens_d_threshold).
    #     Uses scipy.stats.ttest_ind (Welch). More stable on small n; slower.
    selection_mode: Literal["topk", "consensus"] = "topk"
    bootstrap_n: int = 200
    bootstrap_freq_threshold: float = 0.8
    fdr_alpha: float = 0.05
    cohens_d_threshold: float = 0.3


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


class DiagnosticsStage(_Frozen):
    """Intervention side-effect diagnostics: perplexity drift, KL divergence,
    and accuracy on standard side-effect benchmarks (MMLU / HellaSwag / GSM8K).

    Runs last; sidecar artefacts only, never touches existing ones. Disable
    on remote-only backends (openrouter has no logits API) or when the host
    can't afford the dataset downloads.

    Datasets enabled via `corpora`. Each scorer has its own `n_*` cap to
    keep diagnostic cost bounded — paper-style steering side-effect probes
    (CAA, ITI, RepE, Arditi 2024) use 200–500 questions per benchmark; full
    14K MMLU is overkill for a relative damage measurement.

    GSM8K No-CoT mode: short greedy generation + numeric-answer extraction
    (`gsm8k_max_new_tokens`). Paper-faithful 8-shot CoT is deferred — see
    TODO.md ("Diagnostics: CoT GSM8K"). The No-CoT NLL is still informative
    as a function of α even with low absolute accuracy.

    `corpus_n_chars=0` switches WikiText to a built-in synthetic text — used
    by unit tests so they don't reach the HF datasets cache.

    Multi-method sweep. When `compare_methods` is non-empty the stage also
    rebuilds hooks for every (method, alpha) pair in
    `compare_methods × alpha_sweep` from the same VUF directions and SAE
    feature stats, and writes `diagnostics/method_alpha_sweep.parquet`.
    This is independent of `intervention/meta.parquet` and meant to produce
    a clean "method × α × dataset" matrix in one run. Leave empty for the
    default cross-run workflow (one method per run).
    """
    enabled: bool = True
    # Which benchmarks to score. Order is fixed for output stability.
    corpora: list[Literal["wikitext", "mmlu", "hellaswag", "gsm8k"]] = Field(
        default_factory=lambda: ["wikitext", "mmlu", "hellaswag", "gsm8k"]
    )
    corpus_n_chars: int = 300_000
    n_mmlu: int = 200
    n_hellaswag: int = 200
    n_gsm8k: int = 200
    gsm8k_max_new_tokens: int = 32
    kl_max_prompts: int = 100
    # Multi-method matrix (in-run sweep). Empty list ⇒ stage runs per-variant
    # only (reads `intervention/meta.parquet`, original cross-run workflow).
    compare_methods: list[
        Literal["linear_vuf", "sae_emd", "sae_clamp", "sae_projected"]
    ] = Field(default_factory=list)
    alpha_sweep: list[float] = Field(default_factory=list)


class StagesConfig(_Frozen):
    generate: GenerateStage = GenerateStage()
    hidden_states: HiddenStatesStage = HiddenStatesStage()
    categorize: CategorizeStage = CategorizeStage()
    vuf: VUFStage = VUFStage()
    intervene: InterveneStage = InterveneStage()
    detect: DetectStage = DetectStage()
    sae_features: SAEFeaturesStage = SAEFeaturesStage()
    evaluate: EvaluateStage = EvaluateStage()
    diagnostics: DiagnosticsStage = DiagnosticsStage()


class SAEConfig(_Frozen):
    """Where the SAE comes from, plus dimensions for the fake backend.

    provider=fake uses an in-process random linear projection for tests.
    provider=sae_lens loads a pretrained SAE by (release, sae_id). In that
    case d_in/d_latent are ignored — they come from the SAE config. For
    the fake backend, d_in must match the hidden-state dimensionality of
    the generator model (e.g. 896 for Qwen2.5-0.5B; 8 for FakeBackend).

    Per-layer SAE selection (Gemma-Scope / Llama-Scope are trained per layer):
    - `sae_id` — single SAE for the whole run (legacy, for one-layer configs).
    - `sae_id_template` — string with `{layer}` placeholder, expanded per
      requested layer (e.g. `"layer_{layer}/width_16k/canonical"`).
    - `sae_id_overrides` — explicit per-layer mapping; wins over the template,
      and is the only way to express sparse releases like `mistral-7b-res-wg`
      that only ship SAEs for layers 8/16/24.
    Resolution priority (see `models.sae.resolve_sae_id_for_layer`):
    overrides > template > legacy `sae_id`.
    """

    provider: Literal["fake", "sae_lens"] = "fake"
    d_in: int = 8
    d_latent: int = 16
    seed: int = 42
    release: str | None = None
    sae_id: str | None = None
    sae_id_template: str | None = None
    sae_id_overrides: dict[int, str] = Field(default_factory=dict)


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

    def build_run_id(self, timestamp: str, *, user: str | None = None) -> str:
        """Build a human-readable run_id: `<user>__<dataset>__<model>__<ts>__<hash8>`.

        Username prefix attributes runs in shared storage (`/mnt/ssd/sae-muc/runs/`
        on caniculus). `user` defaults to the current OS user; pass explicitly to
        keep run-ids stable across hosts.
        """
        user_part = _slug(user or _current_user())
        return (
            f"{user_part}__{_slug(self.dataset.name)}__{_slug(self.model.name)}"
            f"__{timestamp}__{self.config_hash()[:8]}"
        )


def _current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


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
