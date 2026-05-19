"""Side-effect benchmark scorers used by the `diagnostics` stage.

Three datasets — standard in the steering literature (CAA, ITI, RepE,
Arditi 2024 "Refusal in language models is mediated by a single direction"):

  * **MMLU** (4-way multiple choice) — knowledge.
  * **HellaSwag** (4-way completion) — common-sense.
  * **GSM8K** (open-ended numeric, No-CoT) — arithmetic reasoning.

All three triangulate intervention damage: a hook that lowers Confident
Hallucination Rate (paper Tab.3) may simultaneously degrade reasoning
(GSM8K) more than recall (MMLU). The α-sweep on each dataset is what
the supervisor asked for.

Implementation:

  * MMLU / HellaSwag are scored via likelihood: for each candidate
    continuation we compute teacher-forced NLL of the choice tokens
    (reuses `HFLocalBackend.forward_nll_with_hook`). Argmin = predicted.
  * GSM8K No-CoT does brief greedy generation (`gsm8k_max_new_tokens`),
    parses the first integer / decimal in the output, compares to gold.
    Full 8-shot CoT generation is deferred — see TODO.md.

Loaders return `list[dict]` items so the stage can mock the HF download
in tests by passing pre-built fixtures. `n_samples` is enforced
deterministically (first N, no shuffle — the question set is already
random in the source dataset).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)


# ---------- HF dataset loaders ------------------------------------------------


def load_mmlu(n_samples: int) -> list[dict]:
    """First N MMLU questions (mixed subjects, test split).

    Item shape: {question, choices: [A,B,C,D], answer: int 0..3}.
    Falls back to an empty list on HF unavailability — caller logs/skips.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("cais/mmlu", "all", split="test")
    except Exception as e:  # noqa: BLE001
        log.warning("diagnostics: MMLU load failed (%s); skipping MMLU scorer", e)
        return []
    items: list[dict] = []
    for i, row in enumerate(ds):
        if i >= n_samples:
            break
        items.append({
            "question": row["question"],
            "choices": list(row["choices"]),
            "answer": int(row["answer"]),
        })
    return items


def load_hellaswag(n_samples: int) -> list[dict]:
    """First N HellaSwag validation items (test labels are gated).

    Item shape: {ctx, endings: [4], answer: int 0..3}.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("Rowan/hellaswag", split="validation")
    except Exception as e:  # noqa: BLE001
        log.warning("diagnostics: HellaSwag load failed (%s); skipping", e)
        return []
    items: list[dict] = []
    for i, row in enumerate(ds):
        if i >= n_samples:
            break
        # `label` is a string-digit "0".."3" — convert to int.
        items.append({
            "ctx": row.get("ctx") or row.get("ctx_a", ""),
            "endings": list(row["endings"]),
            "answer": int(row["label"]),
        })
    return items


def load_gsm8k(n_samples: int) -> list[dict]:
    """First N GSM8K test items.

    Item shape: {question, gold_text (full CoT+answer), gold_number: float}.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split="test")
    except Exception as e:  # noqa: BLE001
        log.warning("diagnostics: GSM8K load failed (%s); skipping", e)
        return []
    items: list[dict] = []
    for i, row in enumerate(ds):
        if i >= n_samples:
            break
        items.append({
            "question": row["question"],
            "gold_text": row["answer"],
            "gold_number": _extract_first_number(row["answer"].split("####")[-1]),
        })
    return items


# ---------- prompt builders / parsers -----------------------------------------


_MMLU_TEMPLATE = (
    "The following is a multiple choice question. "
    "Answer with the letter of the correct option.\n\n"
    "Question: {question}\n"
    "A) {a}\n"
    "B) {b}\n"
    "C) {c}\n"
    "D) {d}\n"
    "Answer:"
)


def _mmlu_prompt_and_choices(item: dict) -> tuple[str, list[str], list[str]]:
    choices = item["choices"]
    prompt = _MMLU_TEMPLATE.format(
        question=item["question"],
        a=choices[0], b=choices[1], c=choices[2], d=choices[3],
    )
    # Continuation candidates: " A" / " B" / " C" / " D" (leading space ⇒ they
    # tokenise as a single letter token under most BPE vocabs).
    return prompt, [" A", " B", " C", " D"], ["A", "B", "C", "D"]


def _hellaswag_prompt_and_choices(item: dict) -> tuple[str, list[str], list[str]]:
    stem = item["ctx"].rstrip()
    endings = item["endings"]
    # No leading-space trick here — HellaSwag endings already start with a space
    # in source, but normalise for tokenisers that strip it.
    return stem, [" " + e.lstrip() for e in endings], endings


_GSM8K_TEMPLATE = "Question: {q}\nAnswer:"


def _gsm8k_prompt(item: dict) -> str:
    return _GSM8K_TEMPLATE.format(q=item["question"])


_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def _extract_first_number(text: str) -> float | None:
    """Return the first signed integer/decimal in `text`, or None.

    Tolerates ',' as decimal separator and stray non-digit prefixes.
    """
    if text is None:
        return None
    s = str(text)
    m = _NUMBER_RE.search(s.replace(",", "."))
    if m is None:
        return None
    try:
        return float(m.group(0).replace(",", "."))
    except ValueError:
        return None


# ---------- scorers -----------------------------------------------------------


HookBuilder = Callable[[], tuple[list[int] | None, Any]]
"""A zero-arg builder that returns (hook_layer, hook_fn) for a single run.

`hook_layer` is None and `hook_fn` is None for the baseline (no-hook) pass.
Using a builder (vs passing the pair directly) lets the scorer re-derive
the hook for every dataset cleanly even when called from a sweep loop.
"""


def _mc_nll_choice(
    llm: Any, prompt: str, continuation: str, hook_layer, hook_fn,
) -> float:
    """Sum NLL of `continuation` tokens given `prompt`, under teacher forcing.

    Approximate but cheap: we score `prompt + continuation` jointly and
    treat the full sum_nll as a comparable signal across continuations of
    similar length. For 1-token continuations (MMLU letter, " A".." D")
    this is exact up to the shared-prefix tokens (which cancel in argmin).
    For HellaSwag we additionally length-normalise (mean NLL) so longer
    endings aren't unfairly penalised.
    """
    sum_nll, n_tokens = llm.forward_nll_with_hook(
        prompt + continuation,
        hook_layer=hook_layer, hook_fn=hook_fn,
    )
    return float(sum_nll), int(n_tokens)


def score_mmlu(
    llm: Any,
    items: list[dict],
    *,
    hook_layer=None,
    hook_fn=None,
) -> dict[str, float]:
    """MMLU accuracy + mean NLL of the gold letter under the given hook."""
    if not items:
        return {"accuracy": float("nan"), "mean_nll": float("nan"), "n": 0}
    correct = 0
    gold_nlls: list[float] = []
    for item in items:
        prompt, conts, _letters = _mmlu_prompt_and_choices(item)
        per_choice_nll: list[float] = []
        for cont in conts:
            sum_nll, n_tok = _mc_nll_choice(llm, prompt, cont, hook_layer, hook_fn)
            # `forward_nll_with_hook` returns the full sequence sum_nll. The
            # shared prefix cancels across choices, so argmin on the full sum
            # is equivalent to argmin on the continuation-only NLL. Keep the
            # full sum for monotonicity with the other scorers.
            per_choice_nll.append(sum_nll)
        pred = int(min(range(4), key=lambda i: per_choice_nll[i]))
        gold = int(item["answer"])
        correct += int(pred == gold)
        gold_nlls.append(per_choice_nll[gold])
    return {
        "accuracy": float(correct) / len(items),
        "mean_nll": float(sum(gold_nlls) / len(gold_nlls)),
        "n": int(len(items)),
    }


def score_hellaswag(
    llm: Any,
    items: list[dict],
    *,
    hook_layer=None,
    hook_fn=None,
) -> dict[str, float]:
    """HellaSwag accuracy + length-normalised NLL of the (stem+gold) sequence.

    NB: `mean_nll` averages over the full prompt+continuation, not the gold
    ending in isolation — `forward_nll_with_hook` returns the joint sum_nll
    and we divide by the joint token count. Argmin across choices still picks
    the right continuation (shared stem cancels), and the same quantity is
    computed for baseline vs hooked, so it remains a clean α-drift signal.
    """
    if not items:
        return {"accuracy": float("nan"), "mean_nll": float("nan"), "n": 0}
    correct = 0
    gold_nlls: list[float] = []
    for item in items:
        prompt, conts, _ = _hellaswag_prompt_and_choices(item)
        per_choice_norm: list[float] = []
        per_choice_raw: list[float] = []
        for cont in conts:
            sum_nll, n_tok = _mc_nll_choice(llm, prompt, cont, hook_layer, hook_fn)
            denom = max(1, n_tok)
            per_choice_norm.append(sum_nll / denom)
            per_choice_raw.append(sum_nll)
        pred = int(min(range(4), key=lambda i: per_choice_norm[i]))
        gold = int(item["answer"])
        correct += int(pred == gold)
        # Report length-normalised NLL for the gold ending (comparable across
        # items with very different ending lengths).
        denom_gold = max(1, len(conts[gold].split()))
        gold_nlls.append(per_choice_norm[gold])
    return {
        "accuracy": float(correct) / len(items),
        "mean_nll": float(sum(gold_nlls) / len(gold_nlls)),
        "n": int(len(items)),
    }


def score_gsm8k(
    llm: Any,
    items: list[dict],
    *,
    hook_layer=None,
    hook_fn=None,
    max_new_tokens: int = 32,
    seed: int = 0,
) -> dict[str, float]:
    """GSM8K No-CoT: short greedy generation, parse first number, compare to gold.

    Accuracy is intentionally low without CoT (5–15% on small models) — the
    important signal is the *delta* between α=0 and α≠0, not the absolute.
    `mean_nll` is the per-token NLL of the `prompt + gold_text` joint
    sequence (same caveat as `score_hellaswag`: not continuation-only, but
    consistent across α and hence a valid drift proxy).
    """
    if not items:
        return {
            "accuracy": float("nan"),
            "mean_nll": float("nan"),
            "n": 0,
        }
    correct = 0
    nlls: list[float] = []
    for item in items:
        prompt = _gsm8k_prompt(item)
        # Brief greedy generation under the hook.
        if hook_layer is not None and hook_fn is not None:
            gens = llm.generate_with_hook(
                [prompt], hook_layer=hook_layer, hook_fn=hook_fn,
                temperature=0.0, max_new_tokens=max_new_tokens, n=1,
                seed=seed,
            )
        else:
            gens = llm.generate(
                [prompt], temperature=0.0, max_new_tokens=max_new_tokens, n=1,
                seed=seed,
            )
        out_text = gens[0][0].text
        pred = _extract_first_number(out_text)
        gold = item.get("gold_number")
        if pred is not None and gold is not None and abs(pred - gold) < 1e-6:
            correct += 1

        # Teacher-forced NLL on the gold continuation, for the ppl-vs-α plot.
        gold_text = item.get("gold_text", "")
        if gold_text:
            sum_nll, n_tok = llm.forward_nll_with_hook(
                prompt + " " + gold_text,
                hook_layer=hook_layer, hook_fn=hook_fn,
            )
            if n_tok > 0:
                nlls.append(sum_nll / n_tok)

    return {
        "accuracy": float(correct) / len(items),
        "mean_nll": float(sum(nlls) / len(nlls)) if nlls else float("nan"),
        "n": int(len(items)),
    }
