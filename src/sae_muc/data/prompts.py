"""Prompt templates from the paper's Appendix A.1.

`ANSWER_PROMPT` is the neutral answer-generation prompt. `ANSWER_ELICITING_PROMPT`
adds an instruction to hedge when uncertain; the paper uses it when generating
answers whose verbal uncertainty is measured by the LLM-as-judge.
"""

from __future__ import annotations

ANSWER_PROMPT = """Please answer the following question.
Question: {question}
Answer:"""

ANSWER_ELICITING_PROMPT = (
    "Answer the following question using a succinct (at most one sentence) "
    "and full answer. If you are uncertain about your answer to the question, "
    "convey this uncertainty verbally by precisely hedging this answer.\n"
    "Question: {question}\nAnswer:"
)


def format_answer_prompt(question: str, *, eliciting: bool = False) -> str:
    template = ANSWER_ELICITING_PROMPT if eliciting else ANSWER_PROMPT
    return template.format(question=question)
