"""Batching by budget rather than by count.

A fixed batch size is sized for the worst chunk in the corpus or it is wrong.
Thirty-two chunks of 40 tokens is 1,280 tokens through the model; thirty-two of
512 is 16,384, which is 12.8 times the work for the same batch count. The
second number is what decides whether the job survives, and a count-based cap
cannot see it. So a batch closes when either cap is reached: max_items, or
max_tokens summed over the batch.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Protocol, TypeVar

from . import ConfigError

DEFAULT_MAX_ITEMS = 32
DEFAULT_MAX_TOKENS = 8192
CHARS_PER_TOKEN = 4

T = TypeVar("T")


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class CharCounter:
    """Length over a chars-per-token constant.

    Four characters per token is a rule of thumb for English prose. It
    over-counts code and under-counts agglutinative languages, and the cost of
    being wrong is a batch of the wrong size rather than a wrong answer, so the
    pipeline defaults to this and keeps the real tokenizer optional.
    """

    def __init__(self, chars_per_token: int = CHARS_PER_TOKEN) -> None:
        if chars_per_token <= 0:
            raise ConfigError("chars_per_token must be positive")
        self._ratio = chars_per_token

    def count(self, text: str) -> int:
        return max(1, -(-len(text) // self._ratio))


class TokenizerCounter:
    """Real token counts from a Hugging Face tokenizer.

    The tokenizer is a few megabytes and the model is gigabytes, so this can be
    worth loading on its own when the corpus is mixed enough that the character
    estimate drifts.
    """

    def __init__(self, model_name: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - needs the hf extra
            raise ConfigError("transformers is not installed. pip install -e '.[hf]'") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count(self, text: str) -> int:  # pragma: no cover - needs a tokenizer download
        return len(self._tokenizer.encode(text, add_special_tokens=True))


def batch_by_budget(
    items: Iterable[T],
    *,
    text_of: Callable[[T], str],
    counter: TokenCounter,
    max_items: int = DEFAULT_MAX_ITEMS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Iterator[list[T]]:
    """Group items into batches that respect both caps, in input order.

    One item whose own token count exceeds max_tokens is yielded on its own
    rather than dropped or silently truncated. Dropping it loses a chunk of the
    corpus with nothing to show that it happened; truncating it is the
    tokenizer's job, not the scheduler's.
    """
    if max_items <= 0:
        raise ConfigError("max_items must be positive")
    if max_tokens <= 0:
        raise ConfigError("max_tokens must be positive")

    batch: list[T] = []
    tokens = 0
    for item in items:
        size = counter.count(text_of(item))
        if batch and (len(batch) >= max_items or tokens + size > max_tokens):
            yield batch
            batch, tokens = [], 0
        batch.append(item)
        tokens += size
        if size >= max_tokens:
            yield batch
            batch, tokens = [], 0
    if batch:
        yield batch


def build_counter(kind: str, *, model: str) -> TokenCounter:
    if kind == "chars":
        return CharCounter()
    if kind == "tokenizer":
        return TokenizerCounter(model)
    raise ConfigError(f"unknown token counter: {kind}")
