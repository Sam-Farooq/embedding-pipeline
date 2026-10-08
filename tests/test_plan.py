from __future__ import annotations

from embedpipe.corpus import scan
from embedpipe.ids import point_id
from embedpipe.manifest import Manifest
from embedpipe.payload import payload_signature
from embedpipe.plan import Action, build_plan

from .conftest import HASH_MODEL, plan_for, write

OTHER_MODEL = "BAAI/bge-small-en-v1.5@unpinned/d384"


def seeded(corpus, *, model: str = HASH_MODEL, extra: dict[str, str] | None = None) -> Manifest:
    """A manifest that already matches the corpus on disk."""
    manifest = Manifest.empty("corpus")
    signature = payload_signature(extra or {})
    for doc in scan(corpus):
        manifest.put(
            doc.doc_id,
            path=str(doc.path),
            content_hash=doc.content_hash,
            model=model,
            payload_signature=signature,
            chunk_count=len(doc.text.split("\n\n")),
        )
    return manifest


def test_an_empty_manifest_makes_everything_new(corpus):
    plan = plan_for(corpus, Manifest.empty("corpus"))
    assert sorted(plan.actions) == ["a.md", "nested/b.md"]
    assert set(plan.actions.values()) == {Action.NEW}
    assert plan.counts()["embed_docs"] == 2
    assert not plan.is_clean()


def test_a_matching_manifest_plans_no_work(corpus):
    plan = plan_for(corpus, seeded(corpus))
    assert plan.unchanged == ["a.md", "nested/b.md"]
    assert plan.counts() == {
        "embed_docs": 0,
        "embed_chunks": 0,
        "payload_only_docs": 0,
        "unchanged_docs": 2,
        "removed_docs": 0,
        "delete_points": 0,
    }
    assert plan.is_clean()


def test_an_edited_file_is_the_only_one_replanned(corpus):
    manifest = seeded(corpus)
    write(corpus, "a.md", "alpha paragraph one, edited\n\nalpha paragraph two")
    plan = plan_for(corpus, manifest)
    assert plan.actions == {"a.md": Action.CHANGED, "nested/b.md": Action.UNCHANGED}
    assert [item.doc_id for item in plan.embed] == ["a.md"]


def test_a_different_model_invalidates_every_document(corpus):
    plan = plan_for(corpus, seeded(corpus, model=OTHER_MODEL))
    assert set(plan.actions.values()) == {Action.MODEL_CHANGED}
    assert len(plan.embed) == 2
    assert plan.unchanged == []


def test_a_new_payload_field_asks_for_payloads_only(corpus):
    plan = plan_for(corpus, seeded(corpus), extra={"tenant": "acme"})
    assert set(plan.actions.values()) == {Action.PAYLOAD_ONLY}
    assert plan.embed == []
    assert plan.counts()["embed_chunks"] == 0
    # The text still has to be reproduced, since the payload carries it.
    assert all(item.chunks for item in plan.payload_only)


def test_a_changed_payload_value_is_also_payload_only(corpus):
    manifest = seeded(corpus, extra={"tenant": "acme"})
    plan = plan_for(corpus, manifest, extra={"tenant": "globex"})
    assert set(plan.actions.values()) == {Action.PAYLOAD_ONLY}


def test_content_wins_over_a_payload_change(corpus):
    manifest = seeded(corpus)
    write(corpus, "a.md", "rewritten")
    plan = plan_for(corpus, manifest, extra={"tenant": "acme"})
    # Re-embedding rewrites the payload anyway, so the document must not be
    # queued twice.
    assert plan.actions["a.md"] == Action.CHANGED
    assert [item.doc_id for item in plan.payload_only] == ["nested/b.md"]


def test_a_deleted_file_hands_back_its_point_ids(corpus):
    manifest = seeded(corpus)
    (corpus / "a.md").unlink()
    plan = plan_for(corpus, manifest)
    assert plan.removed == ["a.md"]
    assert plan.removed_point_ids == [point_id("a.md", 0), point_id("a.md", 1)]
    assert plan.counts()["delete_points"] == 2


def test_the_glob_decides_what_counts_as_the_corpus(corpus, tmp_path):
    write(corpus, "notes.txt", "not markdown")
    plan = plan_for(corpus, Manifest.empty("corpus"))
    assert "notes.txt" not in plan.actions
    txt_plan = build_plan(
        scan(corpus, "**/*.txt"),
        Manifest.empty("corpus"),
        model=HASH_MODEL,
        payload_signature=payload_signature({}),
    )
    assert list(txt_plan.actions) == ["notes.txt"]
