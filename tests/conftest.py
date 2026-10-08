from __future__ import annotations

from pathlib import Path

import pytest

from embedpipe.corpus import scan
from embedpipe.manifest import Manifest
from embedpipe.payload import payload_signature
from embedpipe.plan import build_plan

HASH_MODEL = "hash-not-a-model@v1/d8"


def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def paragraph(marker: str, length: int) -> str:
    """A paragraph of exactly `length` characters, identifiable by marker."""
    body = f"{marker} " + "word " * length
    return body[:length]


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    write(root, "a.md", "alpha paragraph one\n\nalpha paragraph two")
    write(root, "nested/b.md", "beta paragraph")
    return root


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    return tmp_path / "state" / "manifest.json"


def plan_for(
    corpus: Path,
    manifest: Manifest,
    *,
    model: str = HASH_MODEL,
    extra: dict[str, str] | None = None,
    max_chars: int = 1200,
    overlap: int = 150,
):
    return build_plan(
        scan(corpus),
        manifest,
        model=model,
        payload_signature=payload_signature(extra or {}),
        max_chars=max_chars,
        overlap=overlap,
    )
