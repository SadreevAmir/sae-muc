# Minimal prompts aligned with Meta verbal_uncertainty / sem_uncertainty (no package deps).

UNCERTAINTY_SYSTEM = (
    "Answer the following question using a succinct (at most one sentence) and full answer. "
    "If you are uncertain about your answer to the question, convey this uncertainty linguistically "
    "by precisely hedging this answer. "
)


def sentence_instruction_prefix() -> str:
    """Matches semantic_entropy.utils PROMPTS['sentence'] + question line."""
    return "Please answer the following question.\n"


def make_plain_user_content(question: str) -> str:
    """Как user-часть у uncertainty, но без системного промпта про лингвистический hedge."""
    return f"Question: {question}\nAnswer: "


def make_sentence_user_content(question: str) -> str:
    return f"{sentence_instruction_prefix()}Question: {question}\nAnswer:"
