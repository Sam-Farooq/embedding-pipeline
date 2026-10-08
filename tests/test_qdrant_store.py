"""The Qdrant adapter, checked against a fake client.

Nothing here connects to anything. What is worth testing is the shape of the
requests: the ids, the vector width, the payload, and that a collection is not
created twice.
"""

from __future__ import annotations

from types import SimpleNamespace

from embedpipe.ids import point_id
from embedpipe.store import Point, QdrantStore


class FakeClient:
    def __init__(self, exists: bool = False, pages: list[tuple[list[str], object]] | None = None):
        self._exists = exists
        self._pages = pages or [([], None)]
        self.created: list[tuple[str, int, str]] = []
        self.deleted_collections: list[str] = []
        self.upserts: list[list[object]] = []
        self.payloads: list[tuple[list[str], dict]] = []
        self.deletes: list[list[str]] = []
        self.waits: list[bool] = []
        self.scroll_calls: list[object] = []

    def collection_exists(self, name: str) -> bool:
        return self._exists

    def create_collection(self, name, vectors_config):
        self._exists = True
        self.created.append((name, vectors_config.size, str(vectors_config.distance)))

    def delete_collection(self, name):
        self.deleted_collections.append(name)
        self._exists = False

    def upsert(self, collection_name, wait, points):
        self.waits.append(wait)
        self.upserts.append(list(points))

    def set_payload(self, collection_name, payload, points, wait):
        self.payloads.append((list(points), dict(payload)))

    def delete(self, collection_name, points_selector, wait):
        self.deletes.append(list(points_selector.points))

    def count(self, collection_name, exact):
        return SimpleNamespace(count=41)

    def scroll(self, collection_name, limit, offset, with_payload, with_vectors):
        self.scroll_calls.append((limit, offset, with_payload, with_vectors))
        ids, next_offset = self._pages[len(self.scroll_calls) - 1]
        return [SimpleNamespace(id=i) for i in ids], next_offset


def test_a_missing_collection_is_created_once():
    client = FakeClient(exists=False)
    store = QdrantStore(collection="corpus", client=client)
    store.ensure_collection(384)
    store.ensure_collection(384)
    assert len(client.created) == 1
    assert client.created[0][0] == "corpus"
    assert client.created[0][1] == 384
    assert "Cosine" in client.created[0][2]


def test_an_existing_collection_is_left_alone():
    client = FakeClient(exists=True)
    QdrantStore(collection="corpus", client=client).ensure_collection(384)
    assert client.created == []


def test_recreate_drops_before_creating():
    client = FakeClient(exists=True)
    QdrantStore(collection="corpus", client=client).ensure_collection(8, recreate=True)
    assert client.deleted_collections == ["corpus"]
    assert len(client.created) == 1


def test_upsert_sends_derived_ids_and_waits_for_them():
    client = FakeClient(exists=True)
    store = QdrantStore(collection="corpus", client=client)
    store.upsert(
        [
            Point(id=point_id("a.md", 0), vector=[0.5, 0.5], payload={"doc_id": "a.md"}),
            Point(id=point_id("a.md", 1), vector=[0.1, 0.9], payload={"doc_id": "a.md"}),
        ]
    )
    sent = client.upserts[0]
    assert [p.id for p in sent] == [point_id("a.md", 0), point_id("a.md", 1)]
    assert sent[0].vector == [0.5, 0.5]
    assert sent[0].payload == {"doc_id": "a.md"}
    # wait=False would let the manifest record an upsert the server has not
    # accepted, which is the one thing the manifest must never do.
    assert client.waits == [True]


def test_set_payload_targets_only_the_ids_it_is_given():
    client = FakeClient(exists=True)
    store = QdrantStore(collection="corpus", client=client)
    store.set_payload([point_id("a.md", 0)], {"tenant": "acme"})
    assert client.payloads == [([point_id("a.md", 0)], {"tenant": "acme"})]


def test_delete_wraps_ids_and_skips_an_empty_list():
    client = FakeClient(exists=True)
    store = QdrantStore(collection="corpus", client=client)
    store.delete([])
    assert client.deletes == []
    store.delete([point_id("a.md", 4)])
    assert client.deletes == [[point_id("a.md", 4)]]


def test_ids_pages_through_scroll_until_the_offset_runs_out():
    client = FakeClient(exists=True, pages=[(["p1", "p2"], "cursor"), (["p3"], None)])
    store = QdrantStore(collection="corpus", client=client)
    assert store.ids() == {"p1", "p2", "p3"}
    assert len(client.scroll_calls) == 2
    assert client.scroll_calls[1][1] == "cursor"
    # with_vectors stays off: verify compares ids and never looks at a vector.
    assert client.scroll_calls[0][3] is False


def test_count_is_an_int():
    client = FakeClient(exists=True)
    assert QdrantStore(collection="corpus", client=client).count() == 41
