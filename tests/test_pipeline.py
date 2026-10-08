from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from embedpipe import ConfigError
from embedpipe.embedder import HashEmbedder
from embedpipe.ids import point_id
from embedpipe.manifest import Manifest
from embedpipe.pipeline import check_vector_space, execute
from embedpipe.store import MemoryStore, Point

from .conftest import paragraph, plan_for, write


class FlakyStore(MemoryStore):
    """A store that dies on a chosen upsert, which is what a killed box is."""

    def __init__(self, fail_on_upsert: int | None = None) -> None:
        super().__init__()
        self.fail_on_upsert = fail_on_upsert
        self.attempts = 0

    def upsert(self, points: Sequence[Point]) -> None:
        self.attempts += 1
        if self.attempts == self.fail_on_upsert:
            raise RuntimeError("boom: the box went away")
        super().upsert(points)


class RelabelledEmbedder(HashEmbedder):
    """Same width, different vector space. A model swap that fits the collection."""

    @property
    def fingerprint(self) -> str:
        return "other-model@v2/d8"

    def encode(self, texts: list[str]) -> np.ndarray:
        return -super().encode(texts)


class WrongShapeEmbedder(HashEmbedder):
    def encode(self, texts: list[str]) -> np.ndarray:
        return super().encode(texts)[:, :4]


def run(
    corpus: Path,
    manifest_path: Path,
    store,
    embedder,
    counter,
    *,
    extra: dict[str, str] | None = None,
    max_items: int = 32,
    max_tokens: int = 8192,
    max_chars: int = 1200,
    overlap: int = 150,
    commit_every: int = 1,
    recreate: bool = False,
):
    manifest = Manifest.load(manifest_path, "corpus")
    check_vector_space(manifest, embedder, recreate=recreate)
    plan = plan_for(
        corpus,
        manifest,
        model=embedder.fingerprint,
        extra=extra,
        max_chars=max_chars,
        overlap=overlap,
    )
    return execute(
        plan,
        store=store,
        embedder=embedder,
        manifest=manifest,
        manifest_path=manifest_path,
        counter=counter,
        extra=extra or {},
        max_items=max_items,
        max_tokens=max_tokens,
        commit_every=commit_every,
        recreate=recreate,
    )


def test_a_first_run_writes_every_chunk(corpus, manifest_path, store, embedder, counter):
    report = run(corpus, manifest_path, store, embedder, counter)
    manifest = Manifest.load(manifest_path, "corpus")
    assert report.documents_embedded == 2
    assert report.chunks_embedded == store.count() == manifest.chunk_count
    assert manifest.dim == 8
    assert manifest.model == embedder.fingerprint
    assert store.ids() == set(manifest.all_point_ids())
    assert all(len(p.vector) == 8 for p in store.points.values())


def test_a_second_run_embeds_nothing(corpus, manifest_path, store, counter):
    run(corpus, manifest_path, store, HashEmbedder(dim=8), counter)
    before = store.count()
    fresh = HashEmbedder(dim=8)
    report = run(corpus, manifest_path, store, fresh, counter)
    assert fresh.texts_seen == 0
    assert fresh.calls == 0
    assert report.documents_unchanged == 2
    assert report.batches == 0
    assert store.count() == before


def test_only_the_edited_document_is_re_embedded(corpus, manifest_path, store, counter):
    run(corpus, manifest_path, store, HashEmbedder(dim=8), counter)
    write(corpus, "a.md", "alpha rewritten entirely")
    fresh = HashEmbedder(dim=8)
    report = run(corpus, manifest_path, store, fresh, counter)
    assert report.documents_embedded == 1
    assert fresh.texts_seen == 1
    assert store.points[point_id("a.md", 0)].payload["text"] == "alpha rewritten entirely"


def test_a_deleted_document_takes_its_vectors_with_it(corpus, manifest_path, store, counter):
    run(corpus, manifest_path, store, HashEmbedder(dim=8), counter)
    doomed = set(Manifest.load(manifest_path, "corpus").point_ids_for("a.md"))
    assert doomed

    (corpus / "a.md").unlink()
    report = run(corpus, manifest_path, store, HashEmbedder(dim=8), counter)
    manifest = Manifest.load(manifest_path, "corpus")
    assert report.documents_removed == 1
    assert report.points_deleted == len(doomed)
    assert "a.md" not in manifest.documents
    assert not doomed & store.ids()
    assert store.ids() == set(manifest.all_point_ids())


def test_a_payload_backfill_never_touches_the_model(corpus, manifest_path, store, counter):
    run(corpus, manifest_path, store, HashEmbedder(dim=8), counter)
    vectors_before = {pid: list(p.vector) for pid, p in store.points.items()}

    fresh = HashEmbedder(dim=8)
    report = run(corpus, manifest_path, store, fresh, counter, extra={"tenant": "acme"})
    assert fresh.calls == 0
    assert fresh.texts_seen == 0
    assert report.documents_payload_only == 2
    assert report.payload_points_written == len(vectors_before)
    assert all(p.payload["tenant"] == "acme" for p in store.points.values())
    assert {pid: list(p.vector) for pid, p in store.points.items()} == vectors_before


def test_a_backfill_settles_so_the_next_run_is_clean(corpus, manifest_path, store, counter):
    run(corpus, manifest_path, store, HashEmbedder(dim=8), counter, extra={"tenant": "acme"})
    report = run(
        corpus, manifest_path, store, HashEmbedder(dim=8), counter, extra={"tenant": "acme"}
    )
    # ingested_at moves on every run and must stay out of the signature.
    assert report.documents_unchanged == 2
    assert report.payload_points_written == 0


def test_a_crash_mid_document_does_not_half_commit(tmp_path, manifest_path, counter):
    corpus = tmp_path / "corpus"
    write(corpus, "a.md", paragraph("a1", 150))
    write(corpus, "b.md", "\n\n".join(paragraph(f"b{i}", 150) for i in range(3)))
    store = FlakyStore(fail_on_upsert=3)

    first = HashEmbedder(dim=8)
    with pytest.raises(RuntimeError, match="boom"):
        run(
            corpus, manifest_path, store, first, counter,
            max_items=1, max_chars=200, overlap=0,
        )

    manifest = Manifest.load(manifest_path, "corpus")
    assert first.texts_seen == 3
    assert list(manifest.documents) == ["a.md"]
    assert store.count() == 2

    store.fail_on_upsert = None
    second = HashEmbedder(dim=8)
    report = run(
        corpus, manifest_path, store, second, counter, max_items=1, max_chars=200, overlap=0
    )

    # a.md was committed, so resume re-embeds b.md only, all three chunks of it.
    assert second.texts_seen == 3
    assert report.documents_embedded == 1
    # Four points, not five: b.md chunk zero was overwritten, not duplicated.
    assert store.count() == 4
    assert Manifest.load(manifest_path, "corpus").chunk_count == 4


def test_a_dimension_change_is_refused_before_anything_is_written(
    corpus, manifest_path, store, counter
):
    run(corpus, manifest_path, store, HashEmbedder(dim=8), counter)
    with pytest.raises(ConfigError, match="--recreate"):
        run(corpus, manifest_path, store, HashEmbedder(dim=16), counter)
    assert all(len(p.vector) == 8 for p in store.points.values())


def test_recreate_rebuilds_the_collection_at_the_new_width(
    corpus, manifest_path, store, counter
):
    run(corpus, manifest_path, store, HashEmbedder(dim=8), counter)
    run(corpus, manifest_path, store, HashEmbedder(dim=16), counter, recreate=True)
    manifest = Manifest.load(manifest_path, "corpus")
    assert manifest.dim == 16
    assert store.count() == manifest.chunk_count
    assert all(len(p.vector) == 16 for p in store.points.values())


def test_a_model_change_rewrites_every_vector_in_place(corpus, manifest_path, store, counter):
    run(corpus, manifest_path, store, HashEmbedder(dim=8), counter)
    ids_before = store.ids()
    vectors_before = {pid: list(p.vector) for pid, p in store.points.items()}

    swapped = RelabelledEmbedder(dim=8)
    report = run(corpus, manifest_path, store, swapped, counter)
    assert report.documents_embedded == 2
    assert store.ids() == ids_before
    assert all(list(store.points[pid].vector) != vectors_before[pid] for pid in ids_before)
    assert Manifest.load(manifest_path, "corpus").model == "other-model@v2/d8"


def test_an_empty_document_is_recorded_and_not_replanned(
    tmp_path, manifest_path, store, counter
):
    corpus = tmp_path / "corpus"
    write(corpus, "blank.md", "\n\n   \n")
    report = run(corpus, manifest_path, store, HashEmbedder(dim=8), counter)
    assert report.documents_embedded == 1
    assert report.chunks_embedded == 0
    assert store.count() == 0

    again = run(corpus, manifest_path, store, HashEmbedder(dim=8), counter)
    assert again.documents_unchanged == 1
    assert again.documents_embedded == 0


def test_one_batch_can_carry_chunks_from_several_documents(
    corpus, manifest_path, store, embedder, counter
):
    report = run(corpus, manifest_path, store, embedder, counter, max_items=32)
    assert report.batches == 1
    assert report.chunks_embedded == 2
    assert embedder.calls == 1


def test_commit_every_trades_manifest_writes_for_lost_work(
    tmp_path, manifest_path, store, counter
):
    corpus = tmp_path / "corpus"
    for i in range(6):
        write(corpus, f"d{i}.md", paragraph(f"d{i}", 120))
    eager = run(corpus, manifest_path, store, HashEmbedder(dim=8), counter, max_items=1)
    assert eager.batches == 6
    assert eager.manifest_saves == 7  # one per batch, plus the final flush

    manifest_path.unlink()
    lazy = run(
        corpus, manifest_path, MemoryStore(), HashEmbedder(dim=8), counter,
        max_items=1, commit_every=3,
    )
    assert lazy.batches == 6
    assert lazy.manifest_saves == 3


def test_an_embedder_that_returns_the_wrong_width_is_caught(
    corpus, manifest_path, store, counter
):
    with pytest.raises(ConfigError, match="expected"):
        run(corpus, manifest_path, store, WrongShapeEmbedder(dim=8), counter)


def test_oversize_chunks_are_reported_not_hidden(tmp_path, manifest_path, store, counter):
    corpus = tmp_path / "corpus"
    write(corpus, "wide.md", paragraph("wide", 1100))
    report = run(corpus, manifest_path, store, HashEmbedder(dim=8), counter, max_tokens=100)
    assert report.oversize_chunks == 1
    assert report.chunks_embedded == 1
    assert store.count() == 1


def test_hash_vectors_are_deterministic_and_normalised():
    one = HashEmbedder(dim=16).encode(["alpha", "beta"])
    two = HashEmbedder(dim=16).encode(["alpha", "beta"])
    assert np.array_equal(one, two)
    assert not np.array_equal(one[0], one[1])
    assert np.allclose(np.linalg.norm(one, axis=1), 1.0, atol=1e-6)
