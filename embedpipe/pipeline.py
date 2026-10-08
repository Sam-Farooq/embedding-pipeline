"""Running a plan.

The ordering here is the part that survives a crash:

1. Delete the points of documents that left the corpus, then drop them from the
   manifest. A kill between the two leaves a manifest entry for a document that
   is already gone from both the corpus and the store, and the next run deletes
   ids that are not there, which Qdrant treats as a no-op.
2. Embed in batches that can span documents, because batching per document
   wastes the budget on a corpus of short files.
3. Write a manifest record for a document only once every one of its chunks has
   been upserted. A batch that dies halfway leaves that document absent from the
   manifest, so the next run re-embeds it from chunk zero. Point ids are
   derived, not random, so those re-upserts overwrite rather than duplicate.
4. Payload-only updates last, since they touch no vectors.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import ConfigError
from .batching import DEFAULT_MAX_ITEMS, DEFAULT_MAX_TOKENS, TokenCounter, batch_by_budget
from .embedder import Embedder
from .ids import point_id
from .manifest import Manifest, utc_now
from .payload import build_payload
from .plan import Plan, PlanItem
from .store import Point, VectorStore


@dataclass
class RunReport:
    documents_embedded: int = 0
    chunks_embedded: int = 0
    documents_payload_only: int = 0
    payload_points_written: int = 0
    documents_removed: int = 0
    points_deleted: int = 0
    documents_unchanged: int = 0
    batches: int = 0
    oversize_chunks: int = 0
    manifest_saves: int = 0

    def rows(self) -> list[tuple[str, int]]:
        return [
            ("documents embedded", self.documents_embedded),
            ("chunks embedded", self.chunks_embedded),
            ("batches", self.batches),
            ("chunks over the token budget", self.oversize_chunks),
            ("documents payload-only", self.documents_payload_only),
            ("payload writes", self.payload_points_written),
            ("documents removed", self.documents_removed),
            ("points deleted", self.points_deleted),
            ("documents unchanged", self.documents_unchanged),
            ("manifest saves", self.manifest_saves),
        ]


@dataclass(frozen=True)
class _Unit:
    doc_id: str
    chunk_index: int
    text: str


class _OversizeWatch:
    """Counts items that blow the per-batch token budget on their own.

    batch_by_budget calls count() exactly once per item, so wrapping the counter
    gets this for free. Counting it separately would tokenize the corpus twice.
    """

    def __init__(self, inner: TokenCounter, limit: int) -> None:
        self._inner = inner
        self._limit = limit
        self.oversize = 0

    def count(self, text: str) -> int:
        tokens = self._inner.count(text)
        if tokens > self._limit:
            self.oversize += 1
        return tokens


def check_vector_space(manifest: Manifest, embedder: Embedder, *, recreate: bool) -> None:
    """Refuse to mix dimensions in one collection.

    Qdrant rejects a point of the wrong width, so a model swap to a different
    size fails at the first upsert with every earlier vector still in place and
    the manifest claiming they are current. Better to refuse before writing
    anything and say which flag fixes it.
    """
    if recreate or manifest.dim is None:
        return
    if manifest.dim != embedder.dim:
        raise ConfigError(
            f"manifest records {manifest.dim}-dimensional vectors from {manifest.model}, "
            f"this run would write {embedder.dim} from {embedder.fingerprint}. "
            "Re-run with --recreate to drop the collection and rebuild it."
        )


def execute(
    plan: Plan,
    *,
    store: VectorStore,
    embedder: Embedder,
    manifest: Manifest,
    manifest_path: Path,
    counter: TokenCounter,
    extra: Mapping[str, str],
    max_items: int = DEFAULT_MAX_ITEMS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    commit_every: int = 1,
    recreate: bool = False,
    ingested_at: str | None = None,
) -> RunReport:
    if commit_every < 1:
        raise ConfigError("commit_every must be at least 1")

    report = RunReport(documents_unchanged=len(plan.unchanged))
    stamp = ingested_at or utc_now()
    watch = _OversizeWatch(counter, max_tokens)

    store.ensure_collection(embedder.dim, recreate=recreate)
    manifest.dim = embedder.dim

    saves = 0

    def commit() -> None:
        nonlocal saves
        manifest.save(manifest_path)
        saves += 1

    try:
        if plan.removed:
            store.delete(plan.removed_point_ids)
            report.points_deleted += len(plan.removed_point_ids)
            for doc_id in plan.removed:
                manifest.drop(doc_id)
            report.documents_removed = len(plan.removed)
            commit()

        by_id = {item.doc_id: item for item in plan.embed}
        remaining = {item.doc_id: len(item.chunks) for item in plan.embed}

        # A document that chunks to nothing, an empty file or one holding only
        # whitespace, never reaches a batch. Without this it stays NEW forever
        # and every scheduled run plans the same work again.
        for item in plan.embed:
            if not item.chunks:
                report.documents_embedded += 1
                _finish(item, plan, manifest, store, report, stamp)

        units = [
            _Unit(item.doc_id, index, text)
            for item in plan.embed
            for index, text in enumerate(item.chunks)
        ]

        since_flush = 0
        for batch in batch_by_budget(
            units,
            text_of=lambda unit: unit.text,
            counter=watch,
            max_items=max_items,
            max_tokens=max_tokens,
        ):
            vectors = embedder.encode([unit.text for unit in batch])
            if vectors.shape != (len(batch), embedder.dim):
                raise ConfigError(
                    f"embedder returned {vectors.shape}, expected ({len(batch)}, {embedder.dim})"
                )
            store.upsert(
                [
                    Point(
                        id=point_id(unit.doc_id, unit.chunk_index),
                        vector=[float(value) for value in vectors[row]],
                        payload=_payload_for(by_id[unit.doc_id], unit, extra, stamp),
                    )
                    for row, unit in enumerate(batch)
                ]
            )
            report.batches += 1
            report.chunks_embedded += len(batch)

            finished = []
            for unit in batch:
                remaining[unit.doc_id] -= 1
                if remaining[unit.doc_id] == 0:
                    finished.append(unit.doc_id)
            for doc_id in finished:
                report.documents_embedded += 1
                _finish(by_id[doc_id], plan, manifest, store, report, stamp)

            since_flush += 1
            if finished and since_flush >= commit_every:
                commit()
                since_flush = 0

        for item in plan.payload_only:
            for index, text in enumerate(item.chunks):
                # One request per chunk, because the text and the index differ
                # per point and Qdrant's set_payload applies one payload to the
                # ids it is given. Grouping only the static fields would halve
                # the request count and is not done.
                store.set_payload(
                    [point_id(item.doc_id, index)],
                    _payload_for(item, _Unit(item.doc_id, index, text), extra, stamp),
                )
                report.payload_points_written += 1
            report.documents_payload_only += 1
            _finish(item, plan, manifest, store, report, stamp)
        if plan.payload_only:
            commit()
    finally:
        # Covers an exception. It does not cover the process being killed, which
        # is what commit_every is for.
        manifest.save(manifest_path)
        saves += 1

    report.oversize_chunks = watch.oversize
    report.manifest_saves = saves
    return report


def _payload_for(
    item: PlanItem, unit: _Unit, extra: Mapping[str, str], stamp: str
) -> dict[str, Any]:
    return build_payload(
        doc_id=item.doc_id,
        chunk_index=unit.chunk_index,
        chunk_count=len(item.chunks),
        content_hash=item.doc.content_hash,
        text=unit.text,
        ingested_at=stamp,
        extra=extra,
    )


def _finish(
    item: PlanItem,
    plan: Plan,
    manifest: Manifest,
    store: VectorStore,
    report: RunReport,
    stamp: str,
) -> None:
    manifest.put(
        item.doc_id,
        path=str(item.doc.path),
        content_hash=item.doc.content_hash,
        model=plan.model,
        payload_signature=plan.payload_signature,
        chunk_count=len(item.chunks),
        embedded_at=stamp,
    )
