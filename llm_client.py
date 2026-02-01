"""
LLM client for OpenAI API report generation.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _load_env_from_project(project_dir: str | Path) -> None:
    for d in [Path(project_dir), Path(__file__).resolve().parent, Path.cwd()]:
        env_file = d / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            return


def _get_api_key(project_dir: str | Path) -> str | None:
    _load_env_from_project(project_dir)
    key = os.getenv("OPENAI_API_KEY")
    if key and key.strip() and not key.startswith("sk-your"):
        return key.strip()
    for d in [Path(project_dir), Path(__file__).resolve().parent]:
        env_file = d / ".env"
        if env_file.exists():
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if s.startswith("OPENAI_API_KEY="):
                            k = s.split("=", 1)[1].strip().strip('"\'')
                            if k and not k.startswith("sk-your"):
                                return k
            except Exception:
                pass
    return None


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0


_usage = LLMUsage()


def get_usage() -> LLMUsage:
    return _usage


def generate_report(
    system_prompt: str,
    user_prompt: str,
    env_dir: str | Path,
) -> tuple[str | None, str | None]:
    api_key = _get_api_key(env_dir)
    if not api_key:
        return None, "OPENAI_API_KEY not found. Add it to .env (see .env.example)."

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

    if response.usage:
        _usage.prompt_tokens += response.usage.prompt_tokens or 0
        _usage.completion_tokens += response.usage.completion_tokens or 0
        _usage.total_tokens += response.usage.total_tokens or 0
    _usage.request_count += 1

    return text.strip(), None
