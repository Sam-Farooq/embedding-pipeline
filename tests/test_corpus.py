from __future__ import annotations

from pathlib import Path
from random import Random

import pytest

from embedpipe import ConfigError
from embedpipe.corpus import (
    NAMED_IN_MESSAGE,
    check_decoded,
    content_hash,
    decode,
    describe_replacements,
    lossy,
    scan,
)

# cp1252 0x92 is a right single quotation mark. utf-8 has no such byte on its
# own, so the decode substitutes one U+FFFD for it.
CP1252_APOSTROPHE = "A refund is the customer’s money.\n".encode("cp1252")

# U+FFFD as a valid file may legitimately hold it: three bytes of good utf-8.
ENCODED = "\ufffd".encode()


def test_a_clean_utf8_file_needs_no_replacement(tmp_path: Path):
    text, replaced = decode("café, 15 € a month\n".encode())
    assert text == "café, 15 € a month\n"
    assert replaced == 0


def test_a_cp1252_file_counts_what_the_decode_threw_away():
    text, replaced = decode(CP1252_APOSTROPHE)
    assert replaced == 1
    assert text == "A refund is the customer�s money.\n"


def test_each_invalid_run_counts_once():
    # Two separate bad bytes, then a three-byte run that is one invalid subpart.
    assert decode(b"a\x92b\x93c")[1] == 2
    assert decode(b"a\xf0\x9f\x92b")[1] == 1


def test_a_file_that_really_contains_u_fffd_is_not_a_lossy_decode():
    """U+FFFD is a character like any other and encodes to ef bf bd in utf-8.

    Counting replacement characters in the decoded text alone would report a
    file that legitimately contains one as broken for ever.
    """
    data = "a tofu � block\n".encode()
    text, replaced = decode(data)
    assert "�" in text
    assert replaced == 0


def test_scan_carries_the_count_onto_the_document(tmp_path: Path):
    (tmp_path / "refunds.md").write_bytes(CP1252_APOSTROPHE)
    (tmp_path / "clean.md").write_text("nothing wrong here\n", encoding="utf-8")
    docs = {doc.doc_id: doc for doc in scan(tmp_path)}
    assert docs["refunds.md"].replacements == 1
    assert docs["clean.md"].replacements == 0


def test_check_decoded_passes_a_clean_corpus(tmp_path: Path):
    (tmp_path / "clean.md").write_text("nothing wrong here\n", encoding="utf-8")
    check_decoded(scan(tmp_path))


def test_check_decoded_refuses_and_names_the_files(tmp_path: Path):
    (tmp_path / "refunds.md").write_bytes(CP1252_APOSTROPHE)
    (tmp_path / "clean.md").write_text("nothing wrong here\n", encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        check_decoded(scan(tmp_path))
    message = str(caught.value)
    assert "refunds.md" in message
    assert "clean.md" not in message


def test_the_hash_still_covers_bytes_and_not_the_decoded_text():
    """Deliberate, and the reason the count above exists at all.

    Two files that differ only in an invalid byte decode to the same string, so
    a hash over the text would call a re-encoded file unchanged and never
    re-embed it. The bytes keep them apart.
    """
    one, two = b"a\x92b", b"a\x93b"
    assert decode(one)[0] == decode(two)[0]
    assert content_hash(one) != content_hash(two)


def test_the_count_cannot_go_negative_on_bytes_that_mix_both_cases():
    """The count subtracts, so it has to be proved it never subtracts too much.

    An encoded U+FFFD cannot be swallowed by a preceding broken sequence,
    because ef is not a continuation byte, and that is the property this pins.
    """
    seeds = [
        b"",
        ENCODED,
        ENCODED * 3,
        b"\xf0" + ENCODED,
        b"\x92" + ENCODED + b"\x93",
        ENCODED + b"\xf0\x9f" + ENCODED,
        b"\xe2\x82" + ENCODED + b"ac",
    ]
    random = Random(0)
    for _ in range(500):
        seeds.append(
            b"".join(
                random.choice([ENCODED, bytes([random.randrange(256)]), b"text "])
                for _ in range(random.randrange(12))
            )
        )
    for data in seeds:
        assert decode(data)[1] >= 0, data


def test_the_worst_offender_is_named_first_and_the_tail_is_counted(tmp_path: Path):
    for index in range(NAMED_IN_MESSAGE + 2):
        (tmp_path / f"doc{index}.md").write_bytes(b"\x92" * (index + 1))
    docs = scan(tmp_path)
    assert [doc.doc_id for doc in lossy(docs)][0] == f"doc{NAMED_IN_MESSAGE + 1}.md"

    described = describe_replacements(docs)
    assert described.startswith("28 replacement characters")
    assert f"doc{NAMED_IN_MESSAGE + 1}.md (7)" in described
    assert "and 2 more" in described
    assert "doc0.md" not in described


def test_describe_says_nothing_about_a_corpus_that_decoded_cleanly(tmp_path: Path):
    (tmp_path / "clean.md").write_text("nothing wrong here\n", encoding="utf-8")
    assert describe_replacements(scan(tmp_path)) == ""
    assert lossy(scan(tmp_path)) == []
