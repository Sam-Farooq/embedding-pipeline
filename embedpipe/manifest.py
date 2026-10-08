"""The manifest: what is in the collection, and what produced it.

One JSON file, one record per document. The record holds the content hash, the
model fingerprint and the payload signature, which is the minimum needed to
answer "does this document need work" without touching the vector store.

Point ids are not stored. They are derived from the doc_id and the chunk count
(see ids.py), so the manifest stays small and cannot disagree with itself about
which ids a document owns.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from . import ManifestError
from .ids import point_ids as derive_point_ids

SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class DocRecord(BaseModel):
    path: str
    content_hash: str
    model: str
    payload_signature: str
    chunk_count: int = Field(ge=0)
    embedded_at: str


class Manifest(BaseModel):
    schema_version: int = SCHEMA_VERSION
    collection: str
    model: str | None = None
    dim: int | None = None
    updated_at: str | None = None
    documents: dict[str, DocRecord] = Field(default_factory=dict)

    @classmethod
    def empty(cls, collection: str) -> Manifest:
        return cls(collection=collection)

    @classmethod
    def load(cls, path: Path, collection: str) -> Manifest:
        """Read the manifest, or start an empty one if the file is absent.

        A file that exists but cannot be parsed raises. Treating a damaged
        manifest as an empty one would re-embed the whole corpus on the next
        scheduled run, which is the expensive failure dressed up as recovery.
        """
        if not path.exists():
            return cls.empty(collection)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ManifestError(f"{path} does not hold a manifest object")
        version = raw.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ManifestError(
                f"{path} is schema version {version!r}, this build writes {SCHEMA_VERSION}. "
                "Delete the manifest and the collection, then run a full rebuild."
            )
        try:
            manifest = cls.model_validate(raw)
        except ValidationError as exc:
            raise ManifestError(f"{path} does not match the manifest schema: {exc}") from exc
        if manifest.collection != collection:
            raise ManifestError(
                f"{path} tracks collection {manifest.collection!r}, this run targets "
                f"{collection!r}. One manifest per collection: point --manifest somewhere else."
            )
        return manifest

    def save(self, path: Path) -> None:
        """Replace the manifest file atomically.

        A plain write that is interrupted halfway leaves truncated JSON, which
        load() then refuses, which means a crash during the save costs a full
        rebuild. Writing a sibling temp file, flushing it to disk and renaming
        over the target means a reader sees either the old file or the new one.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = utc_now()
        body = json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".manifest-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def put(
        self,
        doc_id: str,
        *,
        path: str,
        content_hash: str,
        model: str,
        payload_signature: str,
        chunk_count: int,
        embedded_at: str | None = None,
    ) -> DocRecord:
        record = DocRecord(
            path=path,
            content_hash=content_hash,
            model=model,
            payload_signature=payload_signature,
            chunk_count=chunk_count,
            embedded_at=embedded_at or utc_now(),
        )
        self.documents[doc_id] = record
        self.model = model
        return record

    def drop(self, doc_id: str) -> DocRecord | None:
        return self.documents.pop(doc_id, None)

    def point_ids_for(self, doc_id: str) -> list[str]:
        record = self.documents.get(doc_id)
        if record is None:
            return []
        return derive_point_ids(doc_id, record.chunk_count)

    def all_point_ids(self) -> list[str]:
        out: list[str] = []
        for doc_id in sorted(self.documents):
            out.extend(self.point_ids_for(doc_id))
        return out

    @property
    def chunk_count(self) -> int:
        return sum(record.chunk_count for record in self.documents.values())
