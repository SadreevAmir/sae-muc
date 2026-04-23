from __future__ import annotations

from sae_muc.data import ANSWER_ELICITING_PROMPT, ANSWER_PROMPT, format_answer_prompt


def test_answer_prompt_renders_question():
    out = format_answer_prompt("What is the capital of France?")
    assert "What is the capital of France?" in out
    assert out.startswith("Please answer the following question.")
    assert out.endswith("Answer:")


def test_eliciting_prompt_instructs_hedging():
    out = format_answer_prompt("Who wrote War and Peace?", eliciting=True)
    assert "uncertain" in out
    assert "hedging" in out
    assert "Who wrote War and Peace?" in out


def test_templates_are_distinct():
    assert ANSWER_PROMPT != ANSWER_ELICITING_PROMPT
