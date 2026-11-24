import pytest


@pytest.fixture(autouse=True)
def _no_llm_from_env(monkeypatch):
    """Tests never reach a real LLM: the provider chain is built from env, which is cleared here."""
    for k in ("GITHUB_TOKEN", "EVENTS_LLM_BASE_URL", "EVENTS_LLM_API_KEY", "EVENTS_LLM_MODEL", "EVENTS_LLM_PROVIDERS"):
        monkeypatch.delenv(k, raising=False)
