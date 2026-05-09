"""Probe CherryIn (open.cherryin.net): list models, optionally test one.

Token-conservative by default: just hits /v1/models (no completion calls).
Pass --test <model_id> to run a single 4-token request on that model.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

import httpx  # noqa: E402
from openai import OpenAI  # noqa: E402


def _print_balance(api_key: str) -> None:
    """Hit the OpenAI-compat billing endpoints New-API exposes."""
    base = "https://open.cherryin.net/v1"
    headers = {"Authorization": f"Bearer {api_key}"}
    print("=== Billing ===")
    try:
        sub = httpx.get(f"{base}/dashboard/billing/subscription", headers=headers, timeout=10).json()
        usg = httpx.get(f"{base}/dashboard/billing/usage", headers=headers, timeout=10).json()
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL fetching billing: {type(e).__name__}: {e}")
        return
    hard_limit_usd = float(sub.get("hard_limit_usd") or 0)
    # OpenAI convention: total_usage is in *cents*.
    usage_cents = float(usg.get("total_usage") or 0)
    usage_usd = usage_cents / 100.0
    remaining_usd = hard_limit_usd - usage_usd
    print(f"  hard_limit  : ${hard_limit_usd:,.2f}")
    print(f"  total_usage : {usage_cents:.4f} cents  (≈ ${usage_usd:.4f})")
    print(f"  remaining   : ${remaining_usd:,.2f}")
    if sub.get("has_payment_method") is not None:
        print(f"  payment_method: {sub['has_payment_method']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        metavar="MODEL_ID",
        help="Send ONE tiny chat request to MODEL_ID (max_tokens=4) to verify.",
    )
    parser.add_argument(
        "--filter",
        default="",
        help="Substring to filter the model list by (e.g. 'qwen', 'glm', 'kimi').",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Print key balance / usage via OpenAI-compat billing endpoints.",
    )
    parser.add_argument(
        "--no-models",
        action="store_true",
        help="Skip the /v1/models listing (useful alongside --balance or --test).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("CHERRYIN_API_KEY", "").strip()
    if not api_key:
        print("ERR: CHERRYIN_API_KEY not set in .env", file=sys.stderr)
        return 1

    client = OpenAI(
        base_url="https://open.cherryin.net/v1",
        api_key=api_key,
        timeout=30.0,
    )

    if args.balance:
        _print_balance(api_key)

    if not args.no_models:
        print("\n=== Fetching /v1/models ===")
        try:
            models = client.models.list().data
        except Exception as e:  # noqa: BLE001
            print(f"FAIL: {type(e).__name__}: {e}")
            return 1

        filtered = [m for m in models if args.filter.lower() in m.id.lower()]
        print(f"Total models: {len(models)};  matching filter {args.filter!r}: {len(filtered)}")
        for m in filtered:
            pricing = getattr(m, "pricing", None)
            cost_tag = ""
            if pricing is not None:
                prompt_price = (
                    getattr(pricing, "prompt", None)
                    if not isinstance(pricing, dict)
                    else pricing.get("prompt")
                )
                if prompt_price is not None:
                    cost_tag = f"  ${prompt_price}"
            print(f"  {m.id}{cost_tag}")

    if args.test:
        print(f"\n=== Testing {args.test!r} with max_tokens=4 ===")
        try:
            resp = client.chat.completions.create(
                model=args.test,
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.1,
                max_tokens=4,
                n=1,
            )
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            print(f"OK    response: {text!r}")
            if usage is not None:
                print(f"      usage: {usage}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {type(e).__name__}: {e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
