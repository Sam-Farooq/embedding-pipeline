from __future__ import annotations

import io
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from embedpipe.cli import EXIT_DRIFT, EXIT_OK, EXIT_REFUSED, EXIT_VERIFY, main

from .conftest import write


def invoke(*args: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(list(args), out=out)
    return code, squash(out.getvalue())


def squash(text: str) -> str:
    """Collapse the table padding so assertions are about content, not columns."""
    return re.sub(r"[ \t]+", " ", text)


class Cmd:
    """The flag soup every subcommand needs, in one place."""

    def __init__(self, tmp_path: Path) -> None:
        self.corpus = tmp_path / "corpus"
        self.manifest = tmp_path / "state" / "manifest.json"
        self.points = tmp_path / "points.jsonl"

    def base(self) -> list[str]:
        return ["--manifest", str(self.manifest), "--collection", "corpus"]

    def model(self) -> list[str]:
        return ["--embedder", "hash", "--dim", "8"]

    def store(self) -> list[str]:
        return ["--store", "jsonl", "--jsonl-path", str(self.points)]

    def plan(self, *extra: str) -> tuple[int, str]:
        return invoke("plan", *self.base(), "--corpus", str(self.corpus), *self.model(), *extra)

    def run(self, *extra: str) -> tuple[int, str]:
        return invoke(
            "run", *self.base(), "--corpus", str(self.corpus),
            *self.model(), *self.store(), *extra,
        )

    def backfill(self, *extra: str) -> tuple[int, str]:
        return invoke(
            "backfill", *self.base(), "--corpus", str(self.corpus),
            *self.model(), *self.store(), *extra,
        )

    def status(self) -> tuple[int, str]:
        return invoke("status", *self.base())

    def verify(self) -> tuple[int, str]:
        return invoke("verify", *self.base(), *self.store())

    def report(self, *extra: str) -> tuple[int, str]:
        return invoke(
            "report", *self.base(), "--corpus", str(self.corpus), *self.model(), *extra
        )


@pytest.fixture
def cmd(tmp_path: Path) -> Cmd:
    handle = Cmd(tmp_path)
    write(handle.corpus, "a.md", "alpha one\n\nalpha two")
    write(handle.corpus, "nested/b.md", "beta")
    return handle


def test_plan_counts_the_work_without_writing_anything(cmd):
    code, text = cmd.plan()
    assert code == EXIT_OK
    assert "documents to embed 2" in text
    assert "new a.md" in text
    assert not cmd.manifest.exists()
    assert not cmd.points.exists()


def test_plan_can_fail_a_scheduled_check(cmd):
    code, _ = cmd.plan("--fail-on-drift")
    assert code == EXIT_DRIFT
    assert cmd.run()[0] == EXIT_OK
    assert cmd.plan("--fail-on-drift")[0] == EXIT_OK


def test_run_writes_points_the_manifest_and_a_summary(cmd):
    code, text = cmd.run()
    assert code == EXIT_OK
    assert "documents embedded 2" in text
    lines = [json.loads(line) for line in cmd.points.read_text().splitlines()]
    assert len(lines) == 2
    assert {row["payload"]["doc_id"] for row in lines} == {"a.md", "nested/b.md"}
    assert all(len(row["vector"]) == 8 for row in lines)
    assert json.loads(cmd.manifest.read_text())["dim"] == 8


def test_dry_run_plans_and_stops(cmd):
    code, text = cmd.run("--dry-run")
    assert code == EXIT_OK
    assert "nothing written" in text
    assert not cmd.points.exists()
    assert not cmd.manifest.exists()


def test_status_reads_the_manifest_back(cmd):
    cmd.run()
    code, text = cmd.status()
    assert code == EXIT_OK
    assert "documents 2" in text
    assert "hash-not-a-model@v1/d8" in text
    assert "manifest bytes" in text


def test_verify_agrees_after_a_run_and_complains_after_a_loss(cmd):
    cmd.run()
    assert cmd.verify()[0] == EXIT_OK
    cmd.points.unlink()
    code, text = cmd.verify()
    assert code == EXIT_VERIFY
    assert "missing from store 2" in text


def test_verify_notices_a_point_the_manifest_does_not_know_about(cmd):
    cmd.run()
    with cmd.points.open("a") as handle:
        handle.write(json.dumps({"id": "stowaway", "vector": [0.0] * 8, "payload": {}}) + "\n")
    code, text = cmd.verify()
    assert code == EXIT_VERIFY
    assert "not in manifest 1" in text
    assert "orphan stowaway" in text


def test_report_writes_a_csv(cmd, tmp_path):
    cmd.run()
    out_path = tmp_path / "reports" / "state.csv"
    code, text = cmd.report("--out", str(out_path))
    assert code == EXIT_OK
    assert "2 rows" in text
    body = out_path.read_text().splitlines()
    assert body[0] == "doc_id,status,chunks,content_hash,model,embedded_at"
    assert len(body) == 3


def test_a_malformed_set_flag_is_refused(cmd):
    assert cmd.plan("--set", "tenant")[0] == EXIT_REFUSED


def test_set_may_not_shadow_a_pipeline_field(cmd):
    assert cmd.plan("--set", "text=hello")[0] == EXIT_REFUSED
    assert cmd.plan("--set", "doc_id=x")[0] == EXIT_REFUSED


def test_backfill_refuses_work_that_needs_a_model(cmd):
    code, _ = cmd.backfill("--set", "tenant=acme")
    assert code == EXIT_REFUSED
    assert not cmd.points.exists()


def test_backfill_adds_a_field_without_re_embedding(cmd):
    cmd.run()
    before = [json.loads(line) for line in cmd.points.read_text().splitlines()]
    code, text = cmd.backfill("--set", "tenant=acme")
    assert code == EXIT_OK
    assert "documents payload-only 2" in text
    assert "chunks embedded 0" in text
    rows = [json.loads(line) for line in cmd.points.read_text().splitlines()]
    after = {row["id"]: row for row in rows}
    for row in before:
        assert after[row["id"]]["vector"] == row["vector"]
        assert after[row["id"]]["payload"]["tenant"] == "acme"


def test_a_width_change_is_refused_with_the_flag_that_fixes_it(cmd, capsys):
    cmd.run()
    code = main(
        ["run", *cmd.base(), "--corpus", str(cmd.corpus), "--embedder", "hash",
         "--dim", "16", *cmd.store()]
    )
    assert code == EXIT_REFUSED
    assert "--recreate" in capsys.readouterr().err


def test_a_corrupt_manifest_stops_the_run(cmd):
    cmd.run()
    cmd.manifest.write_text("{not json")
    assert cmd.plan()[0] == EXIT_REFUSED


def test_a_missing_corpus_directory_is_refused(cmd):
    code, _ = invoke(
        "plan", *cmd.base(), "--corpus", str(cmd.corpus / "nope"), "--embedder", "hash"
    )
    assert code == EXIT_REFUSED


def test_a_subcommand_is_required():
    with pytest.raises(SystemExit):
        main([])


def test_the_module_entry_point_runs():
    done = subprocess.run(
        [sys.executable, "-m", "embedpipe", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0
    assert "backfill" in done.stdout
