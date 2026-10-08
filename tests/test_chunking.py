from __future__ import annotations

import pytest

from embedpipe.chunking import split

from .conftest import paragraph


def test_paragraphs_are_packed_until_the_next_one_will_not_fit():
    text = "\n\n".join([paragraph("p1", 60), paragraph("p2", 60), paragraph("p3", 60)])
    chunks = split(text, max_chars=130, overlap=0)
    assert len(chunks) == 2
    assert "p1" in chunks[0] and "p2" in chunks[0]
    assert "p3" in chunks[1]


def test_no_chunk_exceeds_the_limit():
    text = "\n\n".join(paragraph(f"p{i}", 300) for i in range(8))
    for chunk in split(text, max_chars=250, overlap=40):
        assert len(chunk) <= 250


def test_a_long_paragraph_is_windowed_with_real_overlap():
    text = paragraph("solo", 500)
    chunks = split(text, max_chars=200, overlap=50)
    # 500 characters, 200 wide, 150 of stride: windows at 0, 150 and 300.
    assert len(chunks) == 3
    # The tail of one window has to be the head of the next, or a sentence
    # sitting on the boundary is lost from both.
    for first, second in zip(chunks, chunks[1:], strict=False):
        assert first[-50:] == second[:50]


def test_windows_cover_the_whole_paragraph():
    text = paragraph("solo", 453)
    chunks = split(text, max_chars=100, overlap=25)
    rebuilt = chunks[0]
    for chunk in chunks[1:]:
        rebuilt += chunk[25:]
    assert rebuilt == text


def test_overlap_is_not_paid_between_separate_paragraphs():
    text = "\n\n".join([paragraph("p1", 90), paragraph("p2", 90)])
    chunks = split(text, max_chars=100, overlap=40)
    assert len(chunks) == 2
    assert chunks[0][-40:] != chunks[1][:40]


def test_whitespace_only_input_produces_nothing():
    assert split("   \n\n  \n ") == []
    assert split("") == []


def test_splitting_is_stable_across_calls():
    text = "\n\n".join(paragraph(f"p{i}", 140) for i in range(5))
    assert split(text, max_chars=200, overlap=30) == split(text, max_chars=200, overlap=30)


@pytest.mark.parametrize(
    ("max_chars", "overlap"),
    [(0, 0), (-5, 0), (100, 100), (100, 120), (100, -1)],
)
def test_bad_window_arguments_raise(max_chars, overlap):
    with pytest.raises(ValueError):
        split("text", max_chars=max_chars, overlap=overlap)
