"""Deciding what actually needs work.

Six outcomes per document, and the one that earns its keep is payload-only:
same bytes, same model, different payload fields. That case needs the text
re-chunked and the payloads rewritten, and it needs no forward pass at all.
Without it, adding a tenant field to a corpus means paying to embed the corpus
a second time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .chunking import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP, split
from .corpus import SourceDoc
from .ids import point_id
from .manifest import Manifest


class Action(StrEnum):
    NEW = "new"
    CHANGED = "changed"
    MODEL_CHANGED = "model-changed"
    PAYLOAD_ONLY = "payload-only"
    UNCHANGED = "unchanged"
    REMOVED = "removed"


EMBEDDING_ACTIONS = frozenset({Action.NEW, Action.CHANGED, Action.MODEL_CHANGED})


@dataclass(frozen=True)
class PlanItem:
    doc: SourceDoc
    action: Action
    chunks: list[str]
    orphan_point_ids: list[str] = field(default_factory=list)

    @property
    def doc_id(self) -> str:
        return self.doc.doc_id


@dataclass
class Plan:
    model: str
    payload_signature: str
    embed: list[PlanItem] = field(default_factory=list)
    payload_only: list[PlanItem] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    removed_point_ids: list[str] = field(default_factory=list)
    actions: dict[str, Action] = field(default_factory=dict)

    @property
    def pending_chunks(self) -> int:
        return sum(len(item.chunks) for item in self.embed)

    @property
    def orphan_point_ids(self) -> list[str]:
        return [pid for item in self.embed for pid in item.orphan_point_ids]

    def is_clean(self) -> bool:
        return not (self.embed or self.payload_only or self.removed)

    def counts(self) -> dict[str, int]:
        return {
            "embed_docs": len(self.embed),
            "embed_chunks": self.pending_chunks,
            "payload_only_docs": len(self.payload_only),
            "unchanged_docs": len(self.unchanged),
            "removed_docs": len(self.removed),
            "delete_points": len(self.removed_point_ids) + len(self.orphan_point_ids),
        }


def build_plan(
    sources: list[SourceDoc],
    manifest: Manifest,
    *,
    model: str,
    payload_signature: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> Plan:
    plan = Plan(model=model, payload_signature=payload_signature)
    seen: set[str] = set()

    for doc in sources:
        seen.add(doc.doc_id)
        record = manifest.documents.get(doc.doc_id)
        if record is None:
            action = Action.NEW
        elif record.content_hash != doc.content_hash:
            action = Action.CHANGED
        elif record.model != model:
            # A different model means different vector space. Nothing already
            # in the collection for this document is usable, whatever its text.
            action = Action.MODEL_CHANGED
        elif record.payload_signature != payload_signature:
            action = Action.PAYLOAD_ONLY
        else:
            action = Action.UNCHANGED

        plan.actions[doc.doc_id] = action
        if action is Action.UNCHANGED:
            plan.unchanged.append(doc.doc_id)
            continue

        chunks = split(doc.text, max_chars=max_chars, overlap=overlap)
        if action is Action.PAYLOAD_ONLY:
            plan.payload_only.append(PlanItem(doc=doc, action=action, chunks=chunks))
            continue

        was = record.chunk_count if record else 0
        plan.embed.append(
            PlanItem(
                doc=doc,
                action=action,
                chunks=chunks,
                orphan_point_ids=_orphans(doc.doc_id, was, len(chunks)),
            )
        )

    for doc_id in sorted(set(manifest.documents) - seen):
        plan.actions[doc_id] = Action.REMOVED
        plan.removed.append(doc_id)
        plan.removed_point_ids.extend(manifest.point_ids_for(doc_id))

    return plan


def _orphans(doc_id: str, old_count: int, new_count: int) -> list[str]:
    """Point ids the old version of a document owned and the new one does not.

    A document that shrinks from nine chunks to four leaves five points behind.
    They still match queries and they still carry the old text, so a search
    returns content that is no longer in the corpus. Nothing else in the system
    notices, because the collection count still looks plausible.
    """
    return [point_id(doc_id, i) for i in range(new_count, old_count)]
