from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pytest
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

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

os.environ["VECTOR_STORE_PATH"] =  _config["paths"]["vector_store_path"]
os.environ["TEXT_COLLECTION_NAME"] = _config["collections"]["text_collection"]
os.environ["MULTI_VECTOR_STORE_PATH"] =  _config["paths"]["multi_vector_store_path"]
os.environ["MULTI_TEXT_COLLECTION_NAME"] =  _config["collections"]["multi_text_collection"]
os.environ["MULTI_IMAGE_COLLECTION_NAME"] =  _config["collections"]["multi_image_collection"]


class VectorStoreSettings(BaseModel):
    """Configuration for the on-disk Qdrant store this project uses."""

    path: str = Field(default_factory=lambda: os.environ.get("VECTOR_STORE_PATH", "qdrant_audi_vector_store2"))
    text_collection: str = Field(default_factory=lambda: os.environ.get("TEXT_COLLECTION_NAME", "audi_spec_docs2"))

    multi_path: str = Field(default_factory=lambda: os.environ.get("MULTI_VECTOR_STORE_PATH", "qdrant_audi_vector_store2"))
    multi_text_collection: str = Field(default_factory=lambda: os.environ.get("MULTI_TEXT_COLLECTION_NAME", "audi_spec_docs2"))
    multi_image_collection: str = Field(
        default_factory=lambda: os.environ.get("MULTI_IMAGE_COLLECTION_NAME", "audi_spec_docs2_images")
    )

    @property
    def expected_single_collections(self) -> List[str]:
        return [self.text_collection]

    @property
    def expected_multi_collections(self) -> List[str]:
        return [self.multi_text_collection, self.multi_image_collection]


@pytest.fixture(scope="module")
def settings() -> VectorStoreSettings:
    return VectorStoreSettings()


@pytest.fixture(scope="module")
def store_path_exists(settings: VectorStoreSettings) -> bool:
    return Path(settings.path).is_dir() and Path(settings.multi_path).exists()


@pytest.fixture(scope="module")
def client(settings: VectorStoreSettings, store_path_exists: bool):
    """
    """
    if not store_path_exists:
        pytest.skip(f"Vector store path '{settings.path}' does not exist - run ingestion first.")

    qdrant_client = QdrantClient(path=settings.path)
    yield qdrant_client
    qdrant_client.close()

class TestVectorStoreExists:
    def test_storage_path_exists_on_disk(self, settings: VectorStoreSettings, store_path_exists: bool) -> None:
        assert store_path_exists, (
            f"Vector store path '{settings.path}' does not exist - has ingestion been run yet?"
        )

class TestCollectionsExist:
    def test_text_collection_exists(self, client: QdrantClient, settings: VectorStoreSettings) -> None:
        assert client.collection_exists(settings.text_collection), (
            f"Expected text/language collection '{settings.text_collection}' not found."
        )

    def test_reports_any_other_collections_at_this_path(
            self, client: QdrantClient, settings: VectorStoreSettings
    ) -> None:
        """

        """
        actual: List[str] = [c.name for c in client.get_collections().collections]

        missing = [name for name in settings.expected_single_collections if name not in actual]
        assert not missing, f"Expected collection(s) missing entirely: {missing}. Found: {actual}"

        unexpected = [name for name in actual if name not in settings.expected_single_collections]
        if unexpected:
            print(f"\nNote: found {len(unexpected)} collection(s) outside this project's expected schema: {unexpected}")


class TestCollectionsAreQueryable:
    """
    """

    def test_text_collection_has_points_and_returns_a_sample(
            self, client: QdrantClient, settings: VectorStoreSettings
    ) -> None:
        if not client.collection_exists(settings.text_collection):
            pytest.skip(f"'{settings.text_collection}' does not exist.")

        info = client.get_collection(settings.text_collection)
        assert info.points_count and info.points_count > 0, (
            f"Text collection '{settings.text_collection}' exists but has 0 points."
        )

        points, _ = client.scroll(
            collection_name=settings.text_collection, limit=1, with_payload=True
        )
        assert len(points) == 1

        payload = points[0].payload
        assert "page_content" in payload, (
            f"Expected 'page_content' in payload. Got keys: {list(payload.keys())}"
        )
        assert "metadata" in payload, (
            f"Expected 'metadata' in payload. Got keys: {list(payload.keys())}"
        )
        assert "source" in payload["metadata"], (
            f"Expected 'source' in metadata. Got keys: {list(payload['metadata'].keys())}"
        )
        assert "page" in payload["metadata"], (
            f"Expected 'page' in metadata. Got keys: {list(payload['metadata'].keys())}"
        )


@pytest.fixture(scope="module")
def multi_client(settings: VectorStoreSettings, store_path_exists: bool):
    if not store_path_exists:
        pytest.skip(f"Vector store path '{settings.multi_path}' does not exist - run ingestion first.")

    qdrant_client = QdrantClient(path=settings.multi_path)
    yield qdrant_client
    qdrant_client.close()

class TestMultiVectorStoreExists:
    def test_multi_storage_path_exists_on_disk(self, settings: VectorStoreSettings, store_path_exists: bool) -> None:
        assert store_path_exists, (
            f"Vector store path '{settings.multi_path}' does not exist - has ingestion been run yet?"
        )

class TestMultiCollectionsExist:
    def test_multi_text_collection_exists(self, multi_client: QdrantClient, settings: VectorStoreSettings) -> None:
        assert multi_client.collection_exists(settings.multi_text_collection), (
            f"Expected text/language collection '{settings.multi_text_collection}' not found."
        )

    def test_multi_image_collection_exists(self, multi_client: QdrantClient, settings: VectorStoreSettings) -> None:
        assert multi_client.collection_exists(settings.multi_image_collection), (
            f"Expected multimodal/image collection '{settings.multi_image_collection}' not found "
            "(expected if no images were extracted during ingestion)."
        )

    def test_multi_reports_any_other_collections_at_this_path(
            self, multi_client: QdrantClient, settings: VectorStoreSettings
    ) -> None:

        actual: List[str] = [c.name for c in multi_client.get_collections().collections]

        missing = [name for name in settings.expected_multi_collections if name not in actual]
        assert not missing, f"Expected collection(s) missing entirely: {missing}. Found: {actual}"

        unexpected = [name for name in actual if name not in settings.expected_multi_collections]
        if unexpected:
            print(f"\nNote: found {len(unexpected)} collection(s) outside this project's expected schema: {unexpected}")

class TestMultiCollectionsAreQueryable:
    def test_multi_text_collection_has_points_and_returns_a_sample(
            self, multi_client: QdrantClient, settings: VectorStoreSettings
    ) -> None:
        if not multi_client.collection_exists(settings.multi_text_collection):
            pytest.skip(f"'{settings.multi_text_collection}' does not exist.")

        info = multi_client.get_collection(settings.multi_text_collection)
        assert info.points_count and info.points_count > 0, (
            f"Text collection '{settings.multi_text_collection}' exists but has 0 points."
        )

        points, _ = multi_client.scroll(
            collection_name=settings.multi_text_collection, limit=1, with_payload=True
        )
        assert len(points) == 1

        payload = points[0].payload
        assert "page_content" in payload, (
            f"Expected 'page_content' in payload. Got keys: {list(payload.keys())}"
        )
        assert "metadata" in payload, (
            f"Expected 'metadata' in payload. Got keys: {list(payload.keys())}"
        )
        assert "source" in payload["metadata"], (
            f"Expected 'source' in metadata. Got keys: {list(payload['metadata'].keys())}"
        )
        assert "page" in payload["metadata"], (
            f"Expected 'page' in metadata. Got keys: {list(payload['metadata'].keys())}"
        )

    def test_multi_image_collection_has_points_and_returns_a_sample(
            self, multi_client: QdrantClient, settings: VectorStoreSettings
    ) -> None:
        if not multi_client.collection_exists(settings.multi_image_collection):
            pytest.skip(f"'{settings.multi_image_collection}' does not exist.")

        info = multi_client.get_collection(settings.multi_image_collection)
        if not info.points_count:
            pytest.skip(
                f"'{settings.multi_image_collection}' exists but has 0 points (no images were ingested)."
            )

        points, _ = multi_client.scroll(
            collection_name=settings.multi_image_collection, limit=1, with_payload=True
        )
        assert len(points) == 1

        payload = points[0].payload
        assert "page_content" in payload, (
            f"Expected 'page_content' in payload. Got keys: {list(payload.keys())}"
        )
        assert "metadata" in payload, (
            f"Expected 'metadata' in payload. Got keys: {list(payload.keys())}"
        )
        assert "source" in payload["metadata"], (
            f"Expected 'source' in metadata. Got keys: {list(payload['metadata'].keys())}"
        )