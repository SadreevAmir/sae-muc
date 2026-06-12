"""generate: greedy @ low-T plus N samples @ high-T for every question.

Reads `samples.parquet`, writes `generations.parquet` with one row per
generation. `kind` is `greedy` or `sample`; `gen_idx` enumerates samples
within a question (0 for the single greedy row).

`prompt_kind` ∈ {"plain", "eliciting"} records which Appendix-A.1 prompt
produced the row (paper App A.1 / App C):
  * plain     — "Please answer the following question." Feeds Semantic
                Entropy clustering, the accuracy judge, and the abstention
                "most-likely answer" (App C: "we input a question into the
                language model and sample 10 sequences ...").
  * eliciting — the hedging "...for Verbal Uncertainty" prompt, bound by the
                paper to VU / VUF extraction only (§3.1, App A.1).

`generate.prompt_regime`:
  * "split" (default, paper-faithful) — generate BOTH a plain set (for SU /
    accuracy) and an eliciting set (for VU). Doubles generation cost.
  * "eliciting_only" — generate only the eliciting set (cheaper; for smokes).
    All rows are tagged prompt_kind="eliciting" and every downstream selector
    falls through to them, reproducing the pre-split behaviour.
"""

from __future__ import annotations

import logging

import pandas as pd

from sae_muc.data import format_answer_prompt
from sae_muc.pipeline._utils import PROMPT_ELICITING, PROMPT_PLAIN
from sae_muc.pipeline.context import PipelineContext

log = logging.getLogger(__name__)

INPUT = "samples.parquet"
OUTPUT = "generations.parquet"


def _generate_set(
    ctx: PipelineContext, questions, *, eliciting: bool
) -> tuple[list, list]:
    """Greedy @ T_low + N samples @ T_high for one prompt regime."""
    stage_cfg = ctx.cfg.stages.generate
    prompts = [format_answer_prompt(q, eliciting=eliciting) for q in questions]
    greedy = ctx.llm.generate(
        prompts,
        temperature=stage_cfg.temperature_low,
        max_new_tokens=stage_cfg.max_new_tokens,
        n=1,
        seed=ctx.cfg.seed,
        top_p=stage_cfg.top_p,
        top_k=stage_cfg.top_k,
        batch_size=stage_cfg.batch_size,
    )
    sampled = ctx.llm.generate(
        prompts,
        temperature=stage_cfg.temperature_high,
        max_new_tokens=stage_cfg.max_new_tokens,
        n=stage_cfg.n_samples,
        seed=ctx.cfg.seed,
        top_p=stage_cfg.top_p,
        top_k=stage_cfg.top_k,
        batch_size=stage_cfg.batch_size,
    )
    return greedy, sampled


def _rows_for_set(
    sample_ids, greedy, sampled, *, prompt_kind: str
) -> list[dict]:
    rows: list[dict] = []
    for sample_id, g_list, s_list in zip(sample_ids, greedy, sampled, strict=True):
        rows.append(
            {
                "sample_id": sample_id,
                "kind": "greedy",
                "gen_idx": 0,
                "text": g_list[0].text,
                "finish_reason": g_list[0].finish_reason,
                "prompt_kind": prompt_kind,
            }
        )
        for j, gen in enumerate(s_list):
            rows.append(
                {
                    "sample_id": sample_id,
                    "kind": "sample",
                    "gen_idx": j,
                    "text": gen.text,
                    "finish_reason": gen.finish_reason,
                    "prompt_kind": prompt_kind,
                }
            )
    return rows


def run(ctx: PipelineContext) -> list[str]:
    stage_cfg = ctx.cfg.stages.generate
    samples = ctx.store.load_parquet(INPUT)
    sample_ids = list(samples["sample_id"])
    regime = stage_cfg.prompt_regime

    log.info(
        "generating 1 greedy @T=%.2f + %d samples @T=%.2f per question "
        "(%d questions, regime=%s, top_p=%s, top_k=%s)",
        stage_cfg.temperature_low, stage_cfg.n_samples, stage_cfg.temperature_high,
        len(sample_ids), regime, stage_cfg.top_p, stage_cfg.top_k,
    )

    rows: list[dict] = []
    # Eliciting set drives VU / VUF (always produced).
    elic_greedy, elic_sampled = _generate_set(ctx, samples["question"], eliciting=True)
    rows.extend(_rows_for_set(sample_ids, elic_greedy, elic_sampled, prompt_kind=PROMPT_ELICITING))
    # Plain set drives SU / accuracy / most-likely (paper App C). Skipped in
    # eliciting_only mode, where the eliciting set is reused for everything.
    if regime == "split":
        plain_greedy, plain_sampled = _generate_set(ctx, samples["question"], eliciting=False)
        rows.extend(
            _rows_for_set(sample_ids, plain_greedy, plain_sampled, prompt_kind=PROMPT_PLAIN)
        )

    ctx.store.save_parquet(OUTPUT, pd.DataFrame(rows))
    return [OUTPUT]
