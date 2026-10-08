"""Point ids.

uuid5 over "<doc_id>#<chunk_index>" means the id for a chunk is the same on
every run and on every machine. Two consequences, and both of them matter more
than the cost of computing a hash:

1. An upsert after a crash overwrites the point it wrote last time instead of
   adding a second copy of it, so resume needs no cleanup pass.
2. Deleting a document's vectors needs only its id and its chunk count, which
   the manifest already holds. No secondary index, no filter query.

The price is in chunking.py: the id is tied to the chunk's position, so
inserting a paragraph at the top of a document shifts every later chunk and
invalidates all of them.
"""

from __future__ import annotations

import uuid

NAMESPACE = uuid.UUID("6f6b3e1a-9d7c-5b2f-8a41-1c0d5e7b9a33")


def point_id(doc_id: str, chunk_index: int) -> str:
    if chunk_index < 0:
        raise ValueError("chunk_index must not be negative")
    return str(uuid.uuid5(NAMESPACE, f"{doc_id}#{chunk_index}"))


def point_ids(doc_id: str, chunk_count: int) -> list[str]:
    return [point_id(doc_id, i) for i in range(chunk_count)]
