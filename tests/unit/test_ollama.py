import os
import pytest
import requests

try:
    import tomllib  # stdlib on Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # backport for Python 3.10

CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/config.toml")

with open(CONFIG_PATH, "rb") as f:
    _config = tomllib.load(f)

OLLAMA_GENERATE_URL = os.environ.get("OLLAMA_URL", _config["llm"]["ollama_url"])
OLLAMA_BASE_URL = OLLAMA_GENERATE_URL.split("/api/")[0]
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", _config["llm"]["ollama_model"])

def _ollama_is_reachable() -> bool:
    try:
        requests.get(OLLAMA_BASE_URL, timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_is_reachable(),
    reason=f"Ollama not reachable at {OLLAMA_BASE_URL} - start it with `ollama serve`",
)


def test_ollama_server_is_running():
    response = requests.get(OLLAMA_BASE_URL, timeout=5)
    assert response.status_code == 200


def test_configured_model_is_pulled():
    response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    assert response.status_code == 200

    available_models = [m["name"] for m in response.json().get("models", [])]
    assert any(OLLAMA_MODEL in name for name in available_models), (
        f"Model '{OLLAMA_MODEL}' is not pulled. Available: {available_models}. "
        f"Run: ollama pull {OLLAMA_MODEL}"
    )

def test_generate_endpoint_returns_a_response():
    response = requests.post(
        OLLAMA_GENERATE_URL,
        json={"model": OLLAMA_MODEL, "prompt": "Reply with the single word: pong", "stream": False},
        timeout=60,
    )
    response.raise_for_status()

    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)
    assert data["response"].strip() != ""

