"""OpenRouter probe: list models, find GLM/Kimi/Qwen candidates, run a tiny request."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

from openai import OpenAI  # noqa: E402


api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
if not api_key:
    print("ERR: OPENROUTER_API_KEY not set after loading .env", file=sys.stderr)
    sys.exit(1)

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=60.0)

print("=== Fetching /models ===")
models = client.models.list().data
print(f"Total models listed: {len(models)}")

# Filter: GLM, Kimi / Moonshot, Qwen family (both free and paid)
matches: list[str] = []
for m in models:
    mid = m.id
    low = mid.lower()
    if any(k in low for k in ("glm", "kimi", "moonshot", "z-ai", "zhipu", "thudm")):
        matches.append(mid)
print(f"\nGLM / Kimi / Moonshot candidates ({len(matches)}):")
for mid in matches:
    print(f"  {mid}")


PROMPT = "Question: What is 2 + 2?\nAnswer: 4\nDecisiveness score:"


def _try(model: str) -> bool:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0.1,
            max_tokens=16,
            n=1,
        )
        text = resp.choices[0].message.content or ""
        print(f"  OK    {model}: {text!r}")
        return True
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if len(msg) > 200:
            msg = msg[:200] + "..."
        print(f"  FAIL  {model}: {type(e).__name__}: {msg}")
        return False


print("\n=== Testing GLM / Kimi / Moonshot candidates ===")
for mid in matches:
    _try(mid)
