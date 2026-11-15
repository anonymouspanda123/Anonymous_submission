from . import gemini, mistral, openrouter, deepseek

REGISTRY = {
    "google": gemini,
    "mistral": mistral,
    "openrouter": openrouter,
    "deepseek": deepseek,
}
