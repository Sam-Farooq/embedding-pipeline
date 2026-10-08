"""Deterministic, paragraph-aware splitting.

Chunking runs on every document on every pass, including documents that are
not going to be re-embedded, because it is string work and costs nothing next
to a forward pass. That is what makes a payload backfill cheap: the chunk text
can be reproduced without the model.
"""

from __future__ import annotations

import re

DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP = 150

_PARA = re.compile(r"\n\s*\n")


def split(
    text: str, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP
) -> list[str]:
    """Split text into chunks of at most max_chars characters.

    Paragraphs are packed together until the next one would not fit. A single
    paragraph longer than max_chars is windowed with overlap, because cutting a
    long paragraph on a hard boundary loses the sentence that straddles it.
    Overlap is therefore paid only where it is needed, not between every pair.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be in [0, max_chars)")

    chunks: list[str] = []
    buffer = ""
    for para in (p.strip() for p in _PARA.split(text)):
        if not para:
            continue
        if len(para) > max_chars:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(_window(para, max_chars, overlap))
            continue
        candidate = f"{buffer}\n\n{para}" if buffer else para
        if len(candidate) > max_chars:
            chunks.append(buffer)
            buffer = para
        else:
            buffer = candidate
    if buffer:
        chunks.append(buffer)
    return chunks


def _window(para: str, max_chars: int, overlap: int) -> list[str]:
    step = max_chars - overlap
    pieces: list[str] = []
    start = 0
    while start < len(para):
        pieces.append(para[start : start + max_chars])
        if start + max_chars >= len(para):
            break
        start += step
    return pieces
