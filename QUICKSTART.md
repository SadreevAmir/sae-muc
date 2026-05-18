# Quickstart

Two environments, separate paths:

- **Local** = editor + unit / integration tests on fake backends. No GPU,
  no real LLM weights, no API keys.
- **Server** = real GPU runs. **Docker-only** (no system Python, no host
  venv); see [`scripts/server_setup.md`](scripts/server_setup.md) for the
  detailed walkthrough.

## Prerequisites

- **Local:** Python 3.11+ (3.12 preferred to match the image), git, [uv](https://docs.astral.sh/uv/)
  (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- **Server:** Docker with `nvidia-container-toolkit`, member of the
  `docker` and (on caniculus) `ipadocker` group, NVIDIA GPU with CUDA
  12.2+ driver. HF account with licenses accepted for any gated models
  you want (Gemma-2, Llama-3, Mistral). OpenRouter API key for the judge.

## Local setup (editing + tests)

```bash
git clone https://github.com/SadreevAmir/sae-muc.git && cd sae-muc
uv sync --all-extras          # runtime + dev deps
cp .env.example .env           # tokens NOT needed for local tests
uv run pytest -q               # ~200 unit + integration tests on fakes (~20s)
```

Validate a real-shape config without touching the network or GPU:

```bash
uv run python -c "
from sae_muc.config import load_experiment_config
cfg = load_experiment_config('configs/experiment/gemma2_2b_sae_smoke.yaml')
print(cfg.model.name, cfg.stages.intervene.method)
"
```

For an SAE-availability check against the live sae-lens registry (still
no GPU needed):

```bash
uv run python -c "
from sae_muc.config import load_experiment_config
from sae_muc.models.sae import assert_sae_layers_available
cfg = load_experiment_config('configs/experiment/mistral7b_sae_sparse_smoke.yaml')
assert_sae_layers_available(cfg.sae, [8, 16, 24])
print('OK')
"
```

## Server runs (real GPU)

Full walkthrough at [`scripts/server_setup.md`](scripts/server_setup.md).
TL;DR — first time on the box:

```bash
ssh user@caniculus
git clone https://github.com/SadreevAmir/sae-muc.git ~/sae-muc && cd ~/sae-muc
cp .env.example .env && $EDITOR .env     # HF_TOKEN, OPENROUTER_API_KEY
                                          # (do NOT run huggingface-cli login —
                                          #  it would write the token to shared cache)
scripts/docker/setup_shared.sh           # one-time, idempotent
scripts/docker/build.sh                  # ~5 min first time, per-user image tag
```

Then for every run:

```bash
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
                                          # pick a free L40 / L40S
tmux new -s smoke
scripts/docker/run.sh <gpu> run --config configs/experiment/gemma2_2b_sae_smoke.yaml
                                          # detach: Ctrl-b d
                                          # reattach: tmux attach -t smoke
```

The pipeline writes its full log to `/mnt/ssd/sae-muc/runs/<run_id>/run.log`
**alongside the artefacts**. From any other ssh session:

```bash
tail -f /mnt/ssd/sae-muc/runs/<run_id>/run.log
```

Resume a partially-completed run (your own or a teammate's):

```bash
scripts/docker/run.sh <gpu> run --config <yaml> --run-id <existing_run_id>
# stages with valid manifests are skipped; only what's missing runs
```

## Pull artefacts back to your laptop

```bash
export SAE_MUC_RUNS_REMOTE=user@caniculus:/mnt/ssd/sae-muc/runs
./scripts/sync_artifacts.sh <run_id>            # parquet + json + run.log (small)
./scripts/sync_artifacts.sh --heavy <run_id>    # also safetensors (GBs)
```

(The legacy `SAE_MUC_SSH=user@server:/path/to/sae-muc` env still works
and points at `<repo>/data/runs/`; use it only for ranов that landed in
per-user storage rather than shared.)

## Common operations

- **Switch experiment** — change `--config <yaml>`. Each config + seed
  combination resolves to its own `<run_id>`. Schema reference and
  cookbook recipes live in [`configs/README.md`](configs/README.md).
- **Read a parquet artefact** (locally, after sync, or directly on the
  server through `scripts/docker/shell.sh`):
  ```python
  import pandas as pd
  pd.read_parquet("data/runs/<run_id>/generations.parquet")
  ```
- **Compare before/after intervention** — every run produces
  `metrics_comparison.parquet` with one row per intervention variant.
- **Read a teammate's run** — same path under `/mnt/ssd/sae-muc/runs/`,
  group `ipadocker` makes it readable. Run-id prefix shows the owner.

## Per-category VUF

The paper's `uncertain = {VU ≥ 0.9}` bucket actually mixes two distinct
behaviours: **ABSTAIN** (`"I don't know"`) and **HEDGE** (`"I think X"`).
Both look identical to the decisiveness judge, so the linear VUF averages
them into a single direction whose steering effect is ambiguous (paper
Tab.3 Correctness Rate drops alongside Confident Hallucination Rate —
partly the abstain-half making the model refuse *correct* answers).

The `categorize_uncertainty` stage runs a second LLM-as-judge pass (same
backend as `judge`, new 4-shot prompt) that labels every uncertain-bucket
sample generation as ABSTAIN / HEDGE / CONFIDENT, then majority-votes per
question. When the `vuf.per_category` flag is also on, the `vuf` stage
builds two extra directions against the *shared* certain pool:

  - `r_abstain^(l) = mean(h)_{abstain} − mean(h)_{certain}`
  - `r_hedge^(l)   = mean(h)_{hedge}   − mean(h)_{certain}`

`diagnostics` then writes `diagnostics/category_directions.parquet` with
per-layer cosines. Low `cosine_abstain_hedge` (<~0.7) ⇒ residual stream
encodes them separately and per-category steering is viable; high (>~0.9)
⇒ they share a direction (interesting negative result).

Both flags default off — existing experiment YAMLs produce byte-identical
artefacts. To opt in:

```yaml
stages:
  categorize:
    enabled: true
  vuf:
    per_category: true
```

Quick read — three artefacts answer the disentanglement question from
two angles:

```python
import pandas as pd

# 1. Linear (VUF) view — cosines per layer.
cosines = pd.read_parquet("data/runs/<run_id>/diagnostics/category_directions.parquet")
# columns: layer, cosine_abstain_hedge, cosine_abstain_main, cosine_hedge_main,
#          cosine_cross_main, n_abstain, n_hedge
#   cosine_abstain_hedge ≈ 0 ⇒ ABSTAIN and HEDGE are distinct residual-stream
#       axes → per-category steering viable.
#   cosine_abstain_hedge ≈ 1 ⇒ same axis, paper's lumped VUF is honest.
#   cosine_cross_main ≈ |large| ⇒ paper's main VUF actually encodes the
#       abstain↔hedge type-flip axis, not just uncertain-vs-certain.

# 2. SAE-feature view — Jaccard between top-K feature sets per layer.
jaccards = pd.read_parquet("data/runs/<run_id>/diagnostics/category_sae_jaccards.parquet")
# columns: layer, jaccard_abstain_hedge, jaccard_abstain_main, jaccard_hedge_main,
#          n_main_topk, n_abstain_topk, n_hedge_topk
#   jaccard_abstain_hedge ≈ 0 ⇒ SAE encodes the categories with disjoint
#       feature sets → per-category SAE steering should work.
#   jaccard_abstain_hedge ≈ 1 ⇒ same features, different weights.

# 3. Cross direction (paper VUF axis between two uncertain types) is
#    just one row in category_directions — the cosine_cross_main column —
#    plus a usable safetensors file:
#    `vuf/direction_abstain_vs_hedge_layer_{L}.safetensors`
#    Steering α>0 on this direction flips HEDGE → ABSTAIN (more refusal),
#    α<0 flips ABSTAIN → HEDGE (more substantive answer with caveats).
```

**Cross-checking the two views.** If `cosine_abstain_hedge` is high but
`jaccard_abstain_hedge` is low → linear probes are too coarse, but SAE
sees the structure (publish-worthy result). If both agree (both high or
both low), the answer is unambiguous.

Categories: API cost roughly doubles judge spending on the uncertain bucket
only (certain-bucket questions are skipped — labels there are meaningless).
On a paper-scale TriviaQA run with 1k questions this is ~300-500 extra
judge calls.

## Diagnostics artefacts

The `diagnostics` stage measures intervention *side-effects* on general
LM capability. The paper's Tab.3 only sees QA-specific damage; this
stage adds the standard steering-literature probes (Arditi 2024, CAA,
ITI, RepE):

- **Perplexity** on WikiText-2-raw-v1 (vanilla LM).
- **MMLU** (4-way MC) — knowledge.
- **HellaSwag** (4-way completion) — common-sense.
- **GSM8K** (No-CoT, brief greedy gen + numeric parse) — reasoning.
- **KL divergence** of next-token distributions on cached QA prompts.

Always runs on `--stage all` (toggle via `stages.diagnostics.enabled=false`).
OpenRouter backends short-circuit with a stub — they have no logits API.

```
diagnostics/perplexity.parquet         # variant, mean_ppl, ppl_ratio_vs_baseline, n_tokens
diagnostics/benchmarks.parquet         # variant, {mmlu,hellaswag,gsm8k}_{accuracy,mean_nll,n}
diagnostics/kl.parquet                 # variant, mean_kl_*, top1_disagreement_rate, top5_mass_delta
diagnostics/method_alpha_sweep.parquet # ONLY when compare_methods+alpha_sweep are set
diagnostics/summary.json               # flat one-row-per-variant rollup
```

### Per-variant view (default — cross-run workflow)

Each row in `intervention/meta.parquet` gets one row in each artefact above.
You can compare methods across runs by syncing several `<run_id>`s and
concatenating their `diagnostics/*.parquet` outside the pipeline.

```python
import pandas as pd
ppl   = pd.read_parquet("data/runs/<run_id>/diagnostics/perplexity.parquet")
bench = pd.read_parquet("data/runs/<run_id>/diagnostics/benchmarks.parquet")
kl    = pd.read_parquet("data/runs/<run_id>/diagnostics/kl.parquet")
metrics = pd.read_parquet("data/runs/<run_id>/metrics_comparison.parquet")
# Trade-off: paper Tab.3 win (lower Confident-Hall.-Rate) vs. capability damage.
trade = (
    metrics[["variant", "confident_hallucination_rate"]]
    .merge(ppl[["variant", "ppl_ratio_vs_baseline"]], on="variant")
    .merge(bench[["variant", "mmlu_accuracy", "gsm8k_accuracy"]], on="variant")
    .merge(kl[["variant", "mean_kl_last_prompt_token"]], on="variant")
)
print(trade)
```

### Multi-method × α sweep (single-run matrix)

When you want a "linear_vuf vs sae_emd vs sae_clamp vs sae_projected"
graph without spinning up 4 separate runs, populate both knobs:

```yaml
stages:
  diagnostics:
    compare_methods: [linear_vuf, sae_emd, sae_clamp, sae_projected]
    alpha_sweep:     [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
```

The stage rebuilds hooks for every (method, α) pair from the same VUF
directions + SAE feature stats and writes a long-format
`diagnostics/method_alpha_sweep.parquet`:

```python
import pandas as pd
sweep = pd.read_parquet("data/runs/<run_id>/diagnostics/method_alpha_sweep.parquet")
# columns: method, alpha, dataset, accuracy, mean_nll, layers
# Plot: e.g. accuracy vs alpha, line per method, facet per dataset.
print(sweep.pivot_table(index=["method","alpha"], columns="dataset", values="accuracy"))
```

Note: SAE methods (`sae_emd`, `sae_clamp`) need `sae_features/stats.parquet`,
which only ran historically when `intervene.method` was a SAE method.
Including any SAE entry in `compare_methods` auto-un-gates `sae_features`
on the next `--stage all`.

### Resume a finished run with new diagnostics

```bash
scripts/docker/run.sh <gpu> run --config <yaml> --run-id <existing_run_id> --stage diagnostics
```

### Full knob reference

```yaml
stages:
  diagnostics:
    enabled: true                          # set false to skip entirely
    corpora: [wikitext, mmlu, hellaswag, gsm8k]
    corpus_n_chars: 300_000                # WikiText cap; 0 = synthetic test corpus
    n_mmlu: 200                            # subset size per benchmark
    n_hellaswag: 200
    n_gsm8k: 200
    gsm8k_max_new_tokens: 32               # No-CoT brief generation
    kl_max_prompts: 100
    compare_methods: []                    # [] = per-variant only; non-empty enables the sweep
    alpha_sweep: []
```

## Secrets

`.env` lives per-user inside the repo, gitignored, **never in shared
storage**. The Docker image excludes `.env` from the build context too,
so it's never baked in — it's mounted at runtime via `--env-file`.

Never run `huggingface-cli login` inside the container; it would write
the token to `~/.cache/huggingface/token` which on caniculus resolves to
the shared cache directory, readable by ~50 people.
