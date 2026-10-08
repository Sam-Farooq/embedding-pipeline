"""What goes next to a vector, and how a change to that is detected.

The payload signature covers the field names and the static extras passed with
--set. It deliberately does not cover per-chunk values, and it deliberately
does not cover ingested_at: a timestamp in the signature would mark every
document dirty on every run, which is the whole mechanism defeated by one
field.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

FIELD_NAMES = (
    "doc_id",
    "chunk_index",
    "chunk_count",
    "content_hash",
    "text",
    "ingested_at",
)

RESERVED = frozenset(FIELD_NAMES)


def payload_signature(extra: Mapping[str, str]) -> str:
    material = json.dumps(
        {"fields": sorted(FIELD_NAMES), "extra": dict(sorted(extra.items()))},
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def check_extra(extra: Mapping[str, str]) -> None:
    clash = sorted(RESERVED & set(extra))
    if clash:
        raise ValueError(f"--set may not overwrite pipeline fields: {', '.join(clash)}")


def build_payload(
    *,
    doc_id: str,
    chunk_index: int,
    chunk_count: int,
    content_hash: str,
    text: str,
    ingested_at: str,
    extra: Mapping[str, str],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "content_hash": content_hash,
        "text": text,
        "ingested_at": ingested_at,
    }
    payload.update(extra)
    return payload
