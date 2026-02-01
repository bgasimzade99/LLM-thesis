from dataclasses import dataclass

DEFAULT_MODEL = "llama3"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_UNAVAILABLE_MSG = (
    "Ollama not available. Run: ollama pull llama3 && ollama serve "
    "(or open Ollama app)."
)


@dataclass
class LLMUsage:
    request_count: int = 0


_usage = LLMUsage()


def get_usage() -> LLMUsage:
    return _usage


def check_ollama_available() -> bool:
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def generate_report(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
) -> tuple[str | None, str | None]:
    try:
        import requests
    except ImportError:
        return None, "requests package not installed. Run: pip install requests"

    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    payload = {
        "model": model.strip() or DEFAULT_MODEL,
        "prompt": full_prompt,
        "stream": False,
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    except requests.exceptions.ConnectionError:
        return None, OLLAMA_UNAVAILABLE_MSG
    except requests.exceptions.Timeout:
        return None, "Ollama request timed out. Try a smaller model or shorter prompt."
    except Exception as e:
        return None, f"Ollama error: {str(e)}"

    if r.status_code != 200:
        try:
            err = r.json().get("error", r.text[:200])
        except Exception:
            err = r.text[:200] if r.text else str(r.status_code)
        return None, f"Ollama returned {r.status_code}: {err}"

    try:
        data = r.json()
    except Exception as e:
        return None, f"Invalid Ollama response: {str(e)}"

    text = data.get("response")
    if text is None:
        return None, "Ollama returned empty response"

    _usage.request_count += 1
    return text.strip(), None
