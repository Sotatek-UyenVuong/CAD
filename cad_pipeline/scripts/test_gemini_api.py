"""test_gemini_api.py - Verify Gemini API key + model connectivity.

Usage:
  python -m cad_pipeline.scripts.test_gemini_api
  python -m cad_pipeline.scripts.test_gemini_api --model gemini-2.5-flash
  python -m cad_pipeline.scripts.test_gemini_api --api-key "<KEY>"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv


def _load_dotenv_files() -> None:
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    cad_pipeline_dir = project_root / "cad_pipeline"
    load_dotenv(dotenv_path=cad_pipeline_dir / ".env", override=True)
    load_dotenv(dotenv_path=project_root / ".env", override=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test Gemini API connectivity with configured API key."
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Gemini API key override. If omitted, uses GEMINI_API_KEY from .env.",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model name to call.",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: GEMINI_OK",
        help="Prompt used for test request.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    _load_dotenv_files()
    api_key = (args.api_key or os.getenv("GEMINI_API_KEY", "")).strip()
    if not api_key:
        print("[FAIL] GEMINI_API_KEY is missing.")
        return 1

    try:
        from google import genai
    except Exception as exc:
        print(f"[FAIL] Cannot import google.genai: {exc}")
        return 1

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=args.model,
            contents=args.prompt,
        )
        text = (response.text or "").strip()
        print("[OK] Gemini API request succeeded.")
        print(f"model={args.model}")
        print(f"response={text}")
        return 0
    except Exception as exc:
        print("[FAIL] Gemini API request failed.")
        print(f"model={args.model}")
        print(f"error={exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
