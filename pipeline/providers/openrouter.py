"""
OpenRouter provider.

Calls the OpenRouter API using the openai SDK with a custom base URL.
Supports any model available on OpenRouter (e.g. "google/gemma-3-27b-it",
"anthropic/claude-opus-4", "deepseek/deepseek-r1", etc.).

Configuration
-------------
MODELS   Suggested model IDs available on OpenRouter.
API_KEY  Set via the OPENROUTER_API_KEY environment variable.

Interface
---------
query(model, system_prompt, question_text) -> str
"""

import os

MODELS: list[str] = [
    "google/gemma-3-27b-it",
    "google/gemini-2.5-flash",
    "anthropic/claude-opus-4",
    "deepseek/deepseek-r1",
    "mistralai/mistral-large-2411",
    "meta-llama/llama-3.3-70b-instruct",
    "openai/gpt-oss-120b:free",
]

API_KEY  = os.environ.get("OPEN_AI_API_KEY_PAID", "") or os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1"

# Per-model OpenRouter provider routing preferences (see
# https://openrouter.ai/docs/features/provider-routing).
PROVIDER_PREFERENCES: dict[str, dict] = {
    "openai/gpt-oss-120b": {
        "order": ["deepinfra", "akashml"],
        "allow_fallbacks": True,
        "quantizations": ["bf16"],
    },
}

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        if not API_KEY:
            raise EnvironmentError("OPENROUTER_API_KEY environment variable is not set.")
        _client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )
    return _client


def query(model: str, system_prompt: str, question_text: str) -> str:
    client = _get_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question_text})
    extra_body = {}
    if model in PROVIDER_PREFERENCES:
        extra_body["provider"] = PROVIDER_PREFERENCES[model]
    response = client.chat.completions.create(
        model=model, messages=messages, extra_body=extra_body or None
    )
    return response.choices[0].message.content
