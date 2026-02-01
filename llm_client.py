"""
LLM client for single-shot report generation via OpenAI API.
Logs token usage and fails gracefully if API key is missing.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMUsage:
    """Token usage and request count."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0


_usage = LLMUsage()


def get_usage() -> LLMUsage:
    """Return current token usage stats."""
    return _usage


def generate_report(system_prompt: str, user_prompt: str) -> tuple[str | None, str | None]:
    """
    Single-shot completion. No chat history.
    Returns (report_text, error_message). If success, error_message is None.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip() or api_key.startswith("sk-your"):
        return None, "OPENAI_API_KEY is missing or invalid. Add it to .env (see .env.example)."

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=400,
            temperature=0.2,
        )
    except ImportError:
        return None, "OpenAI package not installed. Run: pip install openai"
    except Exception as e:
        return None, f"API error: {str(e)}"

    choice = response.choices[0] if response.choices else None
    if not choice or not choice.message:
        return None, "Empty response from API"

    text = choice.message.content or ""

    # Log usage
    if response.usage:
        _usage.prompt_tokens += response.usage.prompt_tokens or 0
        _usage.completion_tokens += response.usage.completion_tokens or 0
        _usage.total_tokens += response.usage.total_tokens or 0
    _usage.request_count += 1

    return text.strip(), None
