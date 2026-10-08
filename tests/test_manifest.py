from __future__ import annotations

import json

import pytest

from embedpipe import ManifestError
from embedpipe.ids import point_id
from embedpipe.manifest import SCHEMA_VERSION, Manifest


def _filled(collection: str = "corpus") -> Manifest:
    manifest = Manifest.empty(collection)
    manifest.put(
        "a.md",
        path="/tmp/a.md",
        content_hash="aaaa1111",
        model="hash-not-a-model@v1/d8",
        payload_signature="sig0",
        chunk_count=3,
    )
    manifest.dim = 8
    return manifest


def test_round_trip_through_disk(manifest_path):
    _filled().save(manifest_path)
    loaded = Manifest.load(manifest_path, "corpus")
    assert loaded.documents["a.md"].chunk_count == 3
    assert loaded.documents["a.md"].content_hash == "aaaa1111"
    assert loaded.dim == 8
    assert loaded.model == "hash-not-a-model@v1/d8"


def test_save_creates_the_parent_directory(manifest_path):
    assert not manifest_path.parent.exists()
    _filled().save(manifest_path)
    assert manifest_path.exists()


def test_save_leaves_no_temp_file_behind(manifest_path):
    manifest = _filled()
    manifest.save(manifest_path)
    manifest.save(manifest_path)
    assert [p.name for p in manifest_path.parent.iterdir()] == ["manifest.json"]


def test_a_missing_file_starts_empty(manifest_path):
    manifest = Manifest.load(manifest_path, "corpus")
    assert manifest.documents == {}
    assert manifest.model is None


def test_truncated_json_is_refused_rather_than_treated_as_empty(manifest_path):
    _filled().save(manifest_path)
    body = manifest_path.read_text()
    manifest_path.write_text(body[: len(body) // 2])
    with pytest.raises(ManifestError, match="not valid JSON"):
        Manifest.load(manifest_path, "corpus")


def test_an_older_schema_is_refused_with_instructions(manifest_path):
    _filled().save(manifest_path)
    raw = json.loads(manifest_path.read_text())
    raw["schema_version"] = SCHEMA_VERSION - 1
    manifest_path.write_text(json.dumps(raw))
    with pytest.raises(ManifestError, match="full rebuild"):
        Manifest.load(manifest_path, "corpus")


def test_a_manifest_for_another_collection_is_refused(manifest_path):
    _filled("docs").save(manifest_path)
    with pytest.raises(ManifestError, match="tracks collection"):
        Manifest.load(manifest_path, "corpus")


def test_a_json_array_is_not_a_manifest(manifest_path):
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("[]")
    with pytest.raises(ManifestError, match="manifest object"):
        Manifest.load(manifest_path, "corpus")


def test_point_ids_are_derived_from_the_chunk_count():
    manifest = _filled()
    assert manifest.point_ids_for("a.md") == [point_id("a.md", i) for i in range(3)]
    assert manifest.point_ids_for("missing.md") == []


def test_dropping_a_document_removes_its_ids():
    manifest = _filled()
    assert manifest.chunk_count == 3
    assert manifest.drop("a.md") is not None
    assert manifest.all_point_ids() == []
    assert manifest.chunk_count == 0
    assert manifest.drop("a.md") is None


def test_updated_at_is_stamped_on_save(manifest_path):
    manifest = _filled()
    assert manifest.updated_at is None
    manifest.save(manifest_path)
    assert manifest.updated_at is not None
    assert manifest.updated_at.endswith("+00:00")
