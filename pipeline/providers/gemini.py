"""
Google Gemini provider.

Calls the Google GenAI API (google-genai SDK).
The API client is created once and reused across calls.

Configuration
-------------
MODELS   List of Gemini model IDs (e.g. "gemini-2.5-flash-preview-04-17").
API_KEY  Set via the GOOGLE_API_KEY environment variable.

Interface
---------
query(model, system_prompt, question_text) -> str
"""

import os

MODELS: list[str] = [
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
]
API_KEY = os.environ.get("GOOGLE_API_KEY", "")

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        if not API_KEY:
            raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
        _client = genai.Client(api_key=API_KEY)
    return _client


def query(model: str, system_prompt: str, question_text: str) -> str:
    client = _get_client()
    full_prompt = f"{system_prompt}\n\n{question_text}" if system_prompt else question_text
    response = client.models.generate_content(model=model, contents=full_prompt)
    return response.text
