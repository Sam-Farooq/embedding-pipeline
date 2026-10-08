from __future__ import annotations

from pathlib import Path

import pytest


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
def manifest_path(tmp_path: Path) -> Path:
    return tmp_path / "state" / "manifest.json"
