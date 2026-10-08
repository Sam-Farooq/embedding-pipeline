"""Vector stores behind one interface.

Qdrant is the target. The in-memory and JSONL stores exist so the pipeline can
be exercised with nothing running, which is also how the tests reach the
interesting code without a container.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from . import ConfigError

DEFAULT_COLLECTION = "corpus"
DEFAULT_QDRANT_URL = "http://localhost:6333"


@dataclass
class Point:
    id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    def ensure_collection(self, dim: int, *, recreate: bool = False) -> None: ...

    def upsert(self, points: Sequence[Point]) -> None: ...

    def set_payload(self, point_ids: Sequence[str], payload: Mapping[str, Any]) -> None: ...

    def delete(self, point_ids: Sequence[str]) -> None: ...

    def count(self) -> int: ...

    def ids(self) -> set[str]: ...


class MemoryStore:
    """A dict. Loses everything when the process ends, which is the point."""

    def __init__(self) -> None:
        self.points: dict[str, Point] = {}
        self.upsert_calls = 0
        self.dim: int | None = None

    def ensure_collection(self, dim: int, *, recreate: bool = False) -> None:
        if recreate:
            self.points.clear()
        self.dim = dim

    def upsert(self, points: Sequence[Point]) -> None:
        self.upsert_calls += 1
        for point in points:
            if self.dim is not None and len(point.vector) != self.dim:
                raise ConfigError(
                    f"point {point.id} has {len(point.vector)} dimensions, "
                    f"collection expects {self.dim}"
                )
            self.points[point.id] = point

    def set_payload(self, point_ids: Sequence[str], payload: Mapping[str, Any]) -> None:
        for pid in point_ids:
            existing = self.points.get(pid)
            if existing is None:
                continue
            existing.payload.update(payload)

    def delete(self, point_ids: Sequence[str]) -> None:
        for pid in point_ids:
            self.points.pop(pid, None)

    def count(self) -> int:
        return len(self.points)

    def ids(self) -> set[str]:
        return set(self.points)


class JsonlStore:
    """A local file of points, rewritten on every change.

    A debug sink, not a vector database: there is no index and no search. It
    exists so a run can be inspected with jq and so the offline quickstart in
    the README survives across two invocations.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._points: dict[str, Point] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                raw = json.loads(line)
                self._points[raw["id"]] = Point(
                    id=raw["id"], vector=raw["vector"], payload=raw.get("payload", {})
                )
        self._loaded = True

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps({"id": p.id, "vector": p.vector, "payload": p.payload}, sort_keys=True)
            for _, p in sorted(self._points.items())
        ]
        self.path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def ensure_collection(self, dim: int, *, recreate: bool = False) -> None:
        self._load()
        if recreate:
            self._points.clear()
            self._flush()

    def upsert(self, points: Sequence[Point]) -> None:
        self._load()
        for point in points:
            self._points[point.id] = point
        self._flush()

    def set_payload(self, point_ids: Sequence[str], payload: Mapping[str, Any]) -> None:
        self._load()
        for pid in point_ids:
            if pid in self._points:
                self._points[pid].payload.update(payload)
        self._flush()

    def delete(self, point_ids: Sequence[str]) -> None:
        self._load()
        for pid in point_ids:
            self._points.pop(pid, None)
        self._flush()

    def count(self) -> int:
        self._load()
        return len(self._points)

    def ids(self) -> set[str]:
        self._load()
        return set(self._points)


class QdrantStore:
    """Qdrant over the python client, cosine distance, uuid5 point ids.

    The client is constructed lazily and can be injected, so the request shapes
    this adapter builds are testable without a server.
    """

    def __init__(
        self,
        collection: str = DEFAULT_COLLECTION,
        url: str = DEFAULT_QDRANT_URL,
        client: Any | None = None,
        timeout: int = 30,
    ) -> None:
        self.collection = collection
        self._timeout = timeout
        self._url = url
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:  # pragma: no cover - declared dependency
                raise ConfigError("qdrant-client is not installed") from exc
            self._client = QdrantClient(url=self._url, timeout=self._timeout)
        return self._client

    def ensure_collection(self, dim: int, *, recreate: bool = False) -> None:
        from qdrant_client.models import Distance, VectorParams

        params = VectorParams(size=dim, distance=Distance.COSINE)
        if recreate:
            self.client.delete_collection(self.collection)
            self.client.create_collection(self.collection, vectors_config=params)
            return
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(self.collection, vectors_config=params)

    def upsert(self, points: Sequence[Point]) -> None:
        from qdrant_client.models import PointStruct

        self.client.upsert(
            collection_name=self.collection,
            # wait=True so an upsert that has not been accepted cannot be
            # recorded in the manifest as done. The manifest is the only record
            # of what has been embedded, and it has to be the pessimistic one.
            wait=True,
            points=[
                PointStruct(id=p.id, vector=list(p.vector), payload=dict(p.payload)) for p in points
            ],
        )

    def set_payload(self, point_ids: Sequence[str], payload: Mapping[str, Any]) -> None:
        self.client.set_payload(
            collection_name=self.collection,
            payload=dict(payload),
            points=list(point_ids),
            wait=True,
        )

    def delete(self, point_ids: Sequence[str]) -> None:
        from qdrant_client.models import PointIdsList

        if not point_ids:
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(points=list(point_ids)),
            wait=True,
        )

    def count(self) -> int:
        return int(self.client.count(collection_name=self.collection, exact=True).count)

    def ids(self) -> set[str]:
        """Scroll the whole collection, ids only.

        with_vectors=False matters: the vectors are 384 floats each and verify
        does not look at them, so pulling them would make the check cost more
        than the run it is checking.
        """
        found: set[str] = set()
        offset = None
        while True:
            batch, offset = self.client.scroll(
                collection_name=self.collection,
                limit=1024,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            found.update(str(record.id) for record in batch)
            if offset is None:
                break
        return found


def build_store(kind: str, *, collection: str, url: str, jsonl_path: Path) -> VectorStore:
    if kind == "memory":
        return MemoryStore()
    if kind == "jsonl":
        return JsonlStore(jsonl_path)
    if kind == "qdrant":
        return QdrantStore(collection=collection, url=url)
    raise ConfigError(f"unknown store: {kind}")
