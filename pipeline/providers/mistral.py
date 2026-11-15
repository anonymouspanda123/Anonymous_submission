"""
Mistral provider.

Calls the Mistral API (mistralai SDK).
A client is created once per API key and reused across calls.

Configuration
-------------
MODELS    List of Mistral model IDs (e.g. "mistral-large-2512").
API_KEYS  One or more keys, set via MISTRAL_API_KEY, MISTRAL_API_KEY2,
          MISTRAL_API_KEY3, MISTRAL_API_KEY4, MISTRAL_APL_KEY5,
          MISTRAL_APL_KEY6, MISTRAL_APL_KEY7 (any subset may be present;
          the APL5/6/7 names match separate Mistral organizations).
          Calls rotate round-robin across whichever keys are set, starting
          from a random offset per process, so concurrent subprocesses and
          retries naturally spread load across accounts instead of all
          hammering a single key's rate limit.

Interface
---------
query(model, system_prompt, question_text) -> str

Fallback
--------
If every direct-API attempt for a call fails (e.g. a key is invalid, or the
account's subscription tier doesn't include the requested model — both seen
in practice for mistral-large-2512), the call is retried once via OpenRouter
("mistralai/<model>"), using whichever of OPEN_AI_API_KEY_PAID /
OPENROUTER_API_KEY is set. This keeps the judge label / output directory
tied to "mistral_<model>" regardless of which transport actually served it.
"""

import itertools
import os
import random

MODELS: list[str] = [
    "mistral-large-2512",
    "mistral-small-2503",
    "magistral-medium-2509",
]

_KEY_ENV_VARS = [
    "MISTRAL_API_KEY", "MISTRAL_API_KEY2", "MISTRAL_API_KEY3", "MISTRAL_API_KEY4",
    # MISTRAL_APL_KEY5 excluded: returns 401 Unauthorized, invalid/expired key.
    "MISTRAL_APL_KEY6", "MISTRAL_APL_KEY7",
]
API_KEYS = [v for v in (os.environ.get(name, "") for name in _KEY_ENV_VARS) if v]

_clients: dict[str, object] = {}
_key_cycle = None


def _get_client(key: str):
    if key not in _clients:
        from mistralai.client import Mistral
        _clients[key] = Mistral(api_key=key, timeout_ms=300_000)  # 5 min — magistral reasons slowly
    return _clients[key]


def _next_client():
    global _key_cycle
    if not API_KEYS:
        raise EnvironmentError(
            "No Mistral API key set. Set at least one of: " + ", ".join(_KEY_ENV_VARS)
        )
    if _key_cycle is None:
        start = random.randrange(len(API_KEYS))
        order = API_KEYS[start:] + API_KEYS[:start]
        _key_cycle = itertools.cycle(order)
    return _get_client(next(_key_cycle))


def _extract_content(content):
    if isinstance(content, list):
        # Skip thinking blocks; only join text blocks
        parts = []
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type == "thinking":
                continue
            parts.append(block.text if hasattr(block, "text") else str(block))
        return "".join(parts)
    return content


def _query_direct(model: str, system_prompt: str, question_text: str) -> str:
    client = _next_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question_text})
    response = client.chat.complete(model=model, messages=messages)
    return _extract_content(response.choices[0].message.content)


def query(model: str, system_prompt: str, question_text: str) -> str:
    try:
        return _query_direct(model, system_prompt, question_text)
    except Exception as direct_exc:
        try:
            from providers import openrouter
        except ImportError:
            raise direct_exc
        try:
            return openrouter.query(f"mistralai/{model}", system_prompt, question_text)
        except Exception as fallback_exc:
            raise RuntimeError(
                f"direct Mistral API failed ({direct_exc}); "
                f"OpenRouter fallback also failed ({fallback_exc})"
            ) from fallback_exc
