"""Reading a corpus off disk and hashing it."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from . import ConfigError

HASH_PREFIX_LEN = 16


@dataclass(frozen=True)
class SourceDoc:
    doc_id: str
    path: Path
    text: str
    content_hash: str


def content_hash(data: bytes) -> str:
    """Hash raw bytes, not decoded text.

    Decoding with errors="replace" is lossy, so two files that differ only in
    an invalid byte sequence decode to the same string. Hashing the bytes keeps
    them distinct and the change gets noticed.
    """
    return hashlib.sha256(data).hexdigest()[:HASH_PREFIX_LEN]


def scan(root: Path, glob: str = "**/*.md") -> list[SourceDoc]:
    """Return every matching file under root, ordered by doc_id.

    The order is sorted rather than filesystem order so that batch composition,
    and therefore the point ids written in a given run, do not depend on which
    machine the job ran on.
    """
    if not root.is_dir():
        raise ConfigError(f"corpus directory not found: {root}")
    docs: list[SourceDoc] = []
    for path in sorted(root.glob(glob)):
        if not path.is_file():
            continue
        data = path.read_bytes()
        docs.append(
            SourceDoc(
                doc_id=path.relative_to(root).as_posix(),
                path=path,
                text=data.decode("utf-8", errors="replace"),
                content_hash=content_hash(data),
            )
        )
    return docs
