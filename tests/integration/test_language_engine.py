from __future__ import annotations

import os
import pytest
import requests as http_requests
from fastapi.testclient import TestClient

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/config.toml")

with open(CONFIG_PATH, "rb") as f:
    _config = tomllib.load(f)

OLLAMA_URL = os.environ.get("OLLAMA_URL", _config["llm"]["ollama_url"])
OLLAMA_BASE_URL = OLLAMA_URL.split("/api/")[0]


def _ollama_is_reachable() -> bool:
    try:
        http_requests.get(OLLAMA_BASE_URL, timeout=2)
        return True
    except http_requests.exceptions.RequestException:
        return False


# Fixtures
@pytest.fixture(scope="module")
def client():
    """
    Imports and starts the FastAPI app via TestClient.
    Skips the entire module if the vector store isn't ready
    (prevents confusing import-time crashes).
    """

    from api.language_engine.main import app

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# Health check
class TestHealth:
    def test_health_returns_200(self, client: TestClient) -> None:
        import os

        print(os.getcwd())

        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["status"] == "ok"


# Frontend serving
class TestFrontend:
    def test_home_returns_200(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200

    def test_home_returns_html(self, client: TestClient) -> None:
        response = client.get("/")
        assert "text/html" in response.headers["content-type"]

    def test_home_contains_autospec_title(self, client: TestClient) -> None:
        response = client.get("/")
        assert "AutoSpec Search" in response.text

    def test_favicon_404_is_expected(self, client: TestClient) -> None:
        """Browsers always request /favicon.ico — confirm it 404s cleanly, not 500."""
        response = client.get("/favicon.ico")
        assert response.status_code == 404


# /search — request validation
class TestSearchValidation:
    def test_search_rejects_empty_body(self, client: TestClient) -> None:
        response = client.post("/search", json={})
        assert response.status_code == 422  # Pydantic validation error

    def test_search_rejects_missing_query(self, client: TestClient) -> None:
        response = client.post("/search", json={"top_k": 4})
        assert response.status_code == 422

    def test_search_accepts_query_only(self, client: TestClient) -> None:
        """Minimal valid payload — query is the only required field."""
        response = client.post("/search", json={"query": "wheel torque spec"})
        # 200 means the request was valid; LLM/vector errors are handled inside and return 200 with error text
        assert response.status_code == 200

    def test_search_accepts_optional_fields(self, client: TestClient) -> None:
        response = client.post("/search", json={"query": "brake pad thickness", "top_k": 2, "session_id": None})
        assert response.status_code == 200

    def test_search_rejects_invalid_top_k_type(self, client: TestClient) -> None:
        response = client.post("/search", json={"query": "fuel cap", "top_k": "four"})
        assert response.status_code == 422
