from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pytest
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient


class VectorStoreSettings(BaseModel):
    """Configuration for the on-disk Qdrant store this project uses."""

    path: str = Field(default_factory=lambda: os.environ.get("VECTOR_STORE_PATH", "qdrant_audi_vector_store2"))
    text_collection: str = Field(default_factory=lambda: os.environ.get("TEXT_COLLECTION_NAME", "audi_spec_docs2"))
    image_collection: str = Field(
        default_factory=lambda: os.environ.get("IMAGE_COLLECTION_NAME", "audi_spec_docs2_images")
    )

    @property
    def expected_collections(self) -> List[str]:
        return [self.text_collection, self.image_collection]


@pytest.fixture(scope="module")
def settings() -> VectorStoreSettings:
    return VectorStoreSettings()


@pytest.fixture(scope="module")
def store_path_exists(settings: VectorStoreSettings) -> bool:
    return Path(settings.path).is_dir()


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

    def test_image_collection_exists(self, client: QdrantClient, settings: VectorStoreSettings) -> None:
        assert client.collection_exists(settings.image_collection), (
            f"Expected multimodal/image collection '{settings.image_collection}' not found "
            "(expected if no images were extracted during ingestion)."
        )

    def test_reports_any_other_collections_at_this_path(
            self, client: QdrantClient, settings: VectorStoreSettings
    ) -> None:
        """

        """
        actual: List[str] = [c.name for c in client.get_collections().collections]

        missing = [name for name in settings.expected_collections if name not in actual]
        assert not missing, f"Expected collection(s) missing entirely: {missing}. Found: {actual}"

        unexpected = [name for name in actual if name not in settings.expected_collections]
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

        points, _ = client.scroll(collection_name=settings.text_collection, limit=1, with_payload=True)
        assert len(points) == 1
        assert "content_type" in points[0].payload, (
            "Sample point is missing the expected 'content_type' payload field."
        )

    def test_image_collection_has_points_and_returns_a_sample(
            self, client: QdrantClient, settings: VectorStoreSettings
    ) -> None:
        if not client.collection_exists(settings.image_collection):
            pytest.skip(f"'{settings.image_collection}' does not exist.")

        info = client.get_collection(settings.image_collection)
        if not info.points_count:
            pytest.skip(f"'{settings.image_collection}' exists but has 0 points (no images were ingested).")

        points, _ = client.scroll(collection_name=settings.image_collection, limit=1, with_payload=True)
        assert len(points) == 1
        assert "caption" in points[0].payload, (
            "Sample image point is missing the expected 'caption' payload field."
        )