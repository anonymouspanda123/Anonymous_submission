"""
DeepSeek platform provider.

Calls the DeepSeek API using the openai SDK with DeepSeek's base URL.
See https://platform.deepseek.com for available models and API keys.

Configuration
-------------
MODELS    Suggested model IDs available on DeepSeek.
API_KEY   Set via the DEEPSEEK_API_KEY environment variable.

Interface
---------
query(model, system_prompt, question_text) -> str
"""

import os

MODELS: list[str] = [
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "deepseek-chat",
    "deepseek-reasoner",
]

# Append ":thinking" to a model name to enable thinking mode (e.g. "deepseek-v4-pro:thinking").
THINKING_BUDGET = 8192

# TEMPORARY: DeepSeek's own account balance is depleted (2026-09-16). Routing
# through OpenRouter instead, which has the same deepseek-v4-pro model and a
# funded key. Revert ROUTE_VIA_OPENROUTER to False once DeepSeek is topped up.
ROUTE_VIA_OPENROUTER = True

API_KEY  = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com/v1"

OPENROUTER_API_KEY = os.environ.get("OPEN_ROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        if ROUTE_VIA_OPENROUTER:
            if not OPENROUTER_API_KEY:
                raise EnvironmentError("OPEN_ROUTER_API_KEY environment variable is not set.")
            _client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        else:
            if not API_KEY:
                raise EnvironmentError("DEEPSEEK_API_KEY environment variable is not set.")
            _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _client


def _build_response(model: str, system_prompt: str, question_text: str):
    client = _get_client()

    thinking = model.endswith(":thinking")
    if thinking:
        model = model.removesuffix(":thinking")

    call_model = f"deepseek/{model}" if ROUTE_VIA_OPENROUTER else model

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question_text})

    kwargs = {"model": call_model, "messages": messages}
    if thinking:
        if ROUTE_VIA_OPENROUTER:
            kwargs["extra_body"] = {"reasoning": {"enabled": True}}
        else:
            kwargs["extra_body"] = {"thinking": {"type": "enabled", "budget_tokens": THINKING_BUDGET}}

    return client.chat.completions.create(**kwargs)


def query(model: str, system_prompt: str, question_text: str) -> str:
    return _build_response(model, system_prompt, question_text).choices[0].message.content


def query_verbose(model: str, system_prompt: str, question_text: str) -> tuple[str, str | None]:
    msg = _build_response(model, system_prompt, question_text).choices[0].message
    return msg.content, getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
