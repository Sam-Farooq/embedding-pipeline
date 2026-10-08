"""embedpipe: one entry point, six subcommands, state in a manifest file.

Exit codes, because this runs from cron:

  0  nothing went wrong
  2  refused: bad flags, a manifest that cannot be trusted, a dimension change
  3  plan --fail-on-drift found work to do
  4  verify found the manifest and the store disagreeing
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Sequence
from pathlib import Path

from . import EmbedPipeError
from .batching import DEFAULT_MAX_ITEMS, DEFAULT_MAX_TOKENS, build_counter
from .chunking import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP
from .corpus import scan
from .embedder import DEFAULT_DIM, DEFAULT_MODEL, build_embedder, fingerprint
from .manifest import Manifest
from .payload import check_extra, payload_signature
from .pipeline import check_vector_space, execute
from .plan import Action, Plan, build_plan
from .store import DEFAULT_COLLECTION, DEFAULT_QDRANT_URL, build_store

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_DRIFT = 3
EXIT_VERIFY = 4

DEFAULT_MANIFEST = Path(".embedpipe/manifest.json")
DEFAULT_JSONL = Path("points.jsonl")


def _add_manifest_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)


def _add_corpus_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--glob", default="**/*.md")
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument(
        "--set",
        dest="extra",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="static payload field, repeatable. Changing these triggers a payload backfill.",
    )


def _add_model_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedder", choices=["hash", "sentence-transformers"], required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=None, help="pin the model revision, recorded")
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)


def _add_store_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store", choices=["memory", "jsonl", "qdrant"], default="qdrant")
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--jsonl-path", type=Path, default=DEFAULT_JSONL)


def _add_batch_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-items", type=int, default=DEFAULT_MAX_ITEMS)
    parser.add_argument("--batch-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--token-counter", choices=["chars", "tokenizer"], default="chars")
    parser.add_argument(
        "--commit-every",
        type=int,
        default=1,
        help="flush the manifest after this many batches. Higher means less IO "
        "and more re-embedding if the process is killed.",
    )
    parser.add_argument("--recreate", action="store_true", help="drop the collection first")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embedpipe", description=__doc__.splitlines()[0])
    subs = parser.add_subparsers(dest="command", required=True)

    plan_cmd = subs.add_parser("plan", help="print what a run would do, write nothing")
    _add_manifest_flags(plan_cmd)
    _add_corpus_flags(plan_cmd)
    _add_model_flags(plan_cmd)
    plan_cmd.add_argument(
        "--fail-on-drift", action="store_true", help=f"exit {EXIT_DRIFT} if work is pending"
    )

    run_cmd = subs.add_parser("run", help="embed the delta and upsert it")
    _add_manifest_flags(run_cmd)
    _add_corpus_flags(run_cmd)
    _add_model_flags(run_cmd)
    _add_store_flags(run_cmd)
    _add_batch_flags(run_cmd)
    run_cmd.add_argument("--dry-run", action="store_true")

    backfill_cmd = subs.add_parser("backfill", help="rewrite payloads, embed nothing")
    _add_manifest_flags(backfill_cmd)
    _add_corpus_flags(backfill_cmd)
    _add_model_flags(backfill_cmd)
    _add_store_flags(backfill_cmd)
    _add_batch_flags(backfill_cmd)

    status_cmd = subs.add_parser("status", help="what the manifest says is in the collection")
    _add_manifest_flags(status_cmd)

    report_cmd = subs.add_parser("report", help="per-document state as a table or CSV")
    _add_manifest_flags(report_cmd)
    _add_corpus_flags(report_cmd)
    _add_model_flags(report_cmd)
    report_cmd.add_argument("--out", type=Path, default=None)

    verify_cmd = subs.add_parser("verify", help="compare the manifest against the store")
    _add_manifest_flags(verify_cmd)
    _add_store_flags(verify_cmd)

    return parser


def _parse_extra(pairs: Sequence[str]) -> dict[str, str]:
    extra: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not key or not sep:
            raise ValueError(f"--set wants KEY=VALUE, got {pair!r}")
        extra[key] = value
    check_extra(extra)
    return extra


def _table(rows: Sequence[tuple[str, object]], out) -> None:
    if not rows:
        return
    width = max(len(str(label)) for label, _ in rows)
    for label, value in rows:
        print(f"{str(label).ljust(width)}  {value}", file=out)


def _load(args: argparse.Namespace) -> Manifest:
    return Manifest.load(args.manifest, args.collection)


def _plan_for(args: argparse.Namespace, manifest: Manifest) -> tuple[Plan, dict[str, str]]:
    extra = _parse_extra(args.extra)
    model = fingerprint(args.embedder, model=args.model, revision=args.revision, dim=args.dim)
    plan = build_plan(
        scan(args.corpus, args.glob),
        manifest,
        model=model,
        payload_signature=payload_signature(extra),
        max_chars=args.chunk_chars,
        overlap=args.chunk_overlap,
    )
    return plan, extra


def _plan_rows(plan: Plan) -> list[tuple[str, object]]:
    counts = plan.counts()
    return [
        ("model", plan.model),
        ("payload signature", plan.payload_signature),
        ("documents to embed", counts["embed_docs"]),
        ("chunks to embed", counts["embed_chunks"]),
        ("documents payload-only", counts["payload_only_docs"]),
        ("documents unchanged", counts["unchanged_docs"]),
        ("documents removed", counts["removed_docs"]),
        ("points to delete", counts["delete_points"]),
    ]


def cmd_plan(args: argparse.Namespace, out) -> int:
    plan, _ = _plan_for(args, _load(args))
    _table(_plan_rows(plan), out)
    for doc_id, action in sorted(plan.actions.items()):
        if action is not Action.UNCHANGED:
            print(f"  {action:<14} {doc_id}", file=out)
    if args.fail_on_drift and not plan.is_clean():
        return EXIT_DRIFT
    return EXIT_OK


def _run(args: argparse.Namespace, out, *, payload_only: bool) -> int:
    manifest = _load(args)
    plan, extra = _plan_for(args, manifest)

    if payload_only and (plan.embed or plan.removed):
        print(
            f"backfill refused: {len(plan.embed)} documents need embedding and "
            f"{len(plan.removed)} are gone from the corpus. Use run.",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    if payload_only:
        plan = dataclasses.replace(plan, embed=[], removed=[], removed_point_ids=[])

    embedder = build_embedder(args.embedder, model=args.model, revision=args.revision, dim=args.dim)
    check_vector_space(manifest, embedder, recreate=args.recreate)

    if getattr(args, "dry_run", False):
        _table([*_plan_rows(plan), ("dry run", "nothing written")], out)
        return EXIT_OK

    store = build_store(
        args.store,
        collection=args.collection,
        url=args.qdrant_url,
        jsonl_path=args.jsonl_path,
    )
    report = execute(
        plan,
        store=store,
        embedder=embedder,
        manifest=manifest,
        manifest_path=args.manifest,
        counter=build_counter(args.token_counter, model=args.model),
        extra=extra,
        max_items=args.batch_items,
        max_tokens=args.batch_tokens,
        commit_every=args.commit_every,
        recreate=args.recreate,
    )
    _table(report.rows(), out)
    return EXIT_OK


def cmd_status(args: argparse.Namespace, out) -> int:
    manifest = _load(args)
    size = args.manifest.stat().st_size if args.manifest.exists() else 0
    _table(
        [
            ("manifest", str(args.manifest)),
            ("manifest bytes", size),
            ("collection", manifest.collection),
            ("model", manifest.model or "none yet"),
            ("vector dimensions", manifest.dim or "unknown"),
            ("documents", len(manifest.documents)),
            ("chunks", manifest.chunk_count),
            ("updated", manifest.updated_at or "never"),
        ],
        out,
    )
    return EXIT_OK


def cmd_report(args: argparse.Namespace, out) -> int:
    import pandas as pd

    manifest = _load(args)
    plan, _ = _plan_for(args, manifest)
    columns = ["doc_id", "status", "chunks", "content_hash", "model", "embedded_at"]
    rows = []
    for doc_id in sorted(set(manifest.documents) | set(plan.actions)):
        record = manifest.documents.get(doc_id)
        rows.append(
            {
                "doc_id": doc_id,
                "status": str(plan.actions.get(doc_id, "untracked")),
                "chunks": record.chunk_count if record else 0,
                "content_hash": record.content_hash if record else "",
                "model": record.model if record else "",
                "embedded_at": record.embedded_at if record else "",
            }
        )
    frame = pd.DataFrame(rows, columns=columns).sort_values(["status", "doc_id"])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out, index=False)
        print(f"{len(frame)} rows to {args.out}", file=out)
    else:
        print(frame.to_string(index=False), file=out)
    return EXIT_OK


def cmd_verify(args: argparse.Namespace, out) -> int:
    manifest = _load(args)
    store = build_store(
        args.store,
        collection=args.collection,
        url=args.qdrant_url,
        jsonl_path=args.jsonl_path,
    )
    expected = set(manifest.all_point_ids())
    present = store.ids()
    missing = expected - present
    extra_points = present - expected
    _table(
        [
            ("expected points", len(expected)),
            ("points in store", len(present)),
            ("missing from store", len(missing)),
            ("not in manifest", len(extra_points)),
        ],
        out,
    )
    for pid in sorted(missing)[:10]:
        print(f"  missing {pid}", file=out)
    for pid in sorted(extra_points)[:10]:
        print(f"  orphan  {pid}", file=out)
    return EXIT_VERIFY if (missing or extra_points) else EXIT_OK


def main(argv: Sequence[str] | None = None, out=None) -> int:
    out = out or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            return cmd_plan(args, out)
        if args.command == "run":
            return _run(args, out, payload_only=False)
        if args.command == "backfill":
            return _run(args, out, payload_only=True)
        if args.command == "status":
            return cmd_status(args, out)
        if args.command == "report":
            return cmd_report(args, out)
        if args.command == "verify":
            return cmd_verify(args, out)
    except (EmbedPipeError, ValueError) as exc:
        print(f"embedpipe: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
