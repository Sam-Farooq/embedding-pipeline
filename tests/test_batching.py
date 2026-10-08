from __future__ import annotations

import pytest

from embedpipe import ConfigError
from embedpipe.batching import CharCounter, batch_by_budget


class WordCounter:
    """One token per whitespace-separated word, so tests can state exact sizes."""

    def count(self, text: str) -> int:
        return max(1, len(text.split()))


def batches(texts, **kwargs):
    return list(batch_by_budget(texts, text_of=lambda t: t, counter=WordCounter(), **kwargs))


def test_item_cap_closes_a_batch():
    result = batches(["a"] * 7, max_items=3, max_tokens=1000)
    assert [len(b) for b in result] == [3, 3, 1]


def test_token_cap_closes_a_batch_before_the_item_cap():
    four = "w w w w"
    result = batches([four] * 6, max_items=100, max_tokens=10)
    assert [len(b) for b in result] == [2, 2, 2]


def test_no_batch_exceeds_either_cap():
    texts = ["w " * n for n in range(1, 40)]
    for batch in batches(texts, max_items=5, max_tokens=12):
        assert len(batch) <= 5
        if len(batch) > 1:
            assert sum(WordCounter().count(t) for t in batch) <= 12


def test_an_item_larger_than_the_budget_is_isolated_not_dropped():
    big = "w " * 50
    result = batches(["small", big, "small"], max_items=10, max_tokens=10)
    assert [len(b) for b in result] == [1, 1, 1]
    assert result[1][0] is big


def test_nothing_is_dropped_duplicated_or_reordered():
    texts = [f"t{i} " * ((i % 7) + 1) for i in range(53)]
    flat = [item for batch in batches(texts, max_items=4, max_tokens=9) for item in batch]
    assert flat == texts


def test_empty_input_yields_no_batches():
    assert batches([], max_items=4, max_tokens=9) == []


@pytest.mark.parametrize(("items", "tokens"), [(0, 10), (-1, 10), (4, 0), (4, -3)])
def test_bad_caps_raise(items, tokens):
    with pytest.raises(ConfigError):
        batches(["a"], max_items=items, max_tokens=tokens)


def test_char_counter_rounds_up_and_never_returns_zero():
    counter = CharCounter(chars_per_token=4)
    assert counter.count("") == 1
    assert counter.count("abc") == 1
    assert counter.count("abcd") == 1
    assert counter.count("abcde") == 2
    assert counter.count("x" * 400) == 100


def test_char_counter_rejects_a_zero_ratio():
    with pytest.raises(ConfigError):
        CharCounter(chars_per_token=0)
