"""Reading a corpus off disk, decoding it, and hashing it."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import ConfigError

HASH_PREFIX_LEN = 16

REPLACEMENT = "�"
ENCODED_REPLACEMENT = b"\xef\xbf\xbd"

# How many documents a refusal or a warning names before it stops listing them.
NAMED_IN_MESSAGE = 5


@dataclass(frozen=True)
class SourceDoc:
    doc_id: str
    path: Path
    text: str
    content_hash: str
    replacements: int = 0


def content_hash(data: bytes) -> str:
    """Hash raw bytes, not decoded text.

    Decoding with errors="replace" is lossy, so two files that differ only in
    an invalid byte sequence decode to the same string. Hashing the bytes keeps
    them distinct and the change gets noticed.
    """
    return hashlib.sha256(data).hexdigest()[:HASH_PREFIX_LEN]


def decode(data: bytes) -> tuple[str, int]:
    """Decode as utf-8 with replacement, and count what the replacement cost.

    The count is how many U+FFFD this decode introduced, not how many the result
    holds. U+FFFD is a character a valid file may contain, and it encodes to
    ef bf bd, so the ones already in the bytes are subtracted; otherwise a file
    that legitimately mentions the replacement character would be reported as
    broken on every run for ever.

    One U+FFFD stands for one maximal invalid subpart, so a truncated four-byte
    sequence counts once rather than three times. The number is a count of
    broken runs, which is what names the problem, and not of lost bytes.
    """
    text = data.decode("utf-8", errors="replace")
    return text, text.count(REPLACEMENT) - data.count(ENCODED_REPLACEMENT)


def scan(root: Path, glob: str = "**/*.md") -> list[SourceDoc]:
    """Return every matching file under root, ordered by doc_id.

    The order is sorted rather than filesystem order so that batch composition,
    and therefore the point ids written in a given run, do not depend on which
    machine the job ran on.
    """
    if not root.is_dir():
        raise ConfigError(f"corpus directory not found: {root}")
    docs: list[SourceDoc] = []
    for path in sorted(root.glob(glob)):
        if not path.is_file():
            continue
        data = path.read_bytes()
        text, replacements = decode(data)
        docs.append(
            SourceDoc(
                doc_id=path.relative_to(root).as_posix(),
                path=path,
                text=text,
                content_hash=content_hash(data),
                replacements=replacements,
            )
        )
    return docs


def lossy(docs: Iterable[SourceDoc]) -> list[SourceDoc]:
    """The documents whose decode needed replacement, worst first."""
    return sorted(
        (doc for doc in docs if doc.replacements),
        key=lambda doc: (-doc.replacements, doc.doc_id),
    )


def describe_replacements(docs: Sequence[SourceDoc]) -> str:
    """One line naming the worst offenders, or an empty string when there are none.

    Per document rather than a bare total, because the total says a corpus is
    damaged and the names say which file to convert.
    """
    bad = lossy(docs)
    if not bad:
        return ""
    total = sum(doc.replacements for doc in bad)
    shown = ", ".join(f"{doc.doc_id} ({doc.replacements})" for doc in bad[:NAMED_IN_MESSAGE])
    more = f", and {len(bad) - NAMED_IN_MESSAGE} more" if len(bad) > NAMED_IN_MESSAGE else ""
    return (
        f"{_plural(total, 'replacement character')} from a lossy utf-8 decode, "
        f"in {_plural(len(bad), 'document')} of {len(docs)}: {shown}{more}"
    )


def check_decoded(docs: Sequence[SourceDoc]) -> None:
    """Refuse a corpus that did not decode cleanly.

    The vector is computed from the chunk text, so a file that needed
    replacement is embedded with U+FFFD standing where its bytes were. That is a
    worse answer than an error, because the run succeeds and the counts look
    right.
    """
    described = describe_replacements(docs)
    if described:
        raise ConfigError(
            f"{described}. Those bytes are not utf-8 and the text that would be "
            "embedded carries U+FFFD in their place. Convert the files, or drop "
            "--strict to embed them as they stand."
        )


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"
