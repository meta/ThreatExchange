# Copyright (c) Meta Platforms, Inc. and affiliates.

import pickle

import pytest
from threatexchange.signal_type.index import IndexMatch
from threatexchange.signal_type.raw_text import LevenshteinLinearSearch, RawTextSignal
from threatexchange.signal_type.tests.signal_type_test_helper import (
    MatchesStrAutoTest,
    THashValidateCase,
    TMatchesStrCase,
)
import typing as t


class TestRawTextSignal(MatchesStrAutoTest):
    TYPE = RawTextSignal

    def get_validate_hash_cases(self) -> t.Iterable[THashValidateCase]:
        return [
            ("a", "a"),  # Normal case
            ("a ", "a"),  # Whitespace is trimmed
        ]

    @pytest.mark.skip(
        reason="RawTextSignal is not a FileHasher, compare_hash is not supported"
    )
    def test_compare_hash(self) -> None:
        pass

    def get_matches_str_cases(self) -> t.Iterable[TMatchesStrCase]:
        return [
            ("", "", True, 0),  # Empty strings match
            ("a", "a", True, 0),  # Identical single-character match
            ("a", "b", False, 1),  # Single-character mismatch
            ("aaaaaaaaaa", "a", False, 9),  # Longer string doesn't match shorter
            ("a", "aaaaaaaaa", False, 8),  # Shorter string doesn't match longer
            ("a a a a a a a a a", "aaaaaaaaa", True, 0),  # Normalization removes spaces
            ("a" * 19, "a" * 18 + "b", False, 1),  # Fails threshold 95%
            ("a" * 20, "a" * 19 + "b", True, 1),  # Meets threshold with 20 chars
        ]

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            ("привет мир как дела", "привет мир как дела", True),
            ("東京タワーに行きます", "東京タワーに行きます", True),
            ("привет мир как дела", "пока", False),
        ],
    )
    def test_matches_str_non_ascii(self, a: str, b: str, expected: bool) -> None:
        assert self.TYPE.matches_str(a, b).match == expected

    def test_index_matches_matches_str(self) -> None:
        signals = [
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy cat",
            "привет мир как дела у тебя сегодня",
            "a" * 20,
            "",
        ]
        queries = signals + [
            "the QUICK brown fox jumps over the lazy dogs",
            "a" * 19 + "b",
            "something else entirely",
        ]
        idx = LevenshteinLinearSearch.build((s, i) for i, s in enumerate(signals))
        idx = pickle.loads(pickle.dumps(idx))
        for q in queries:
            expected = {
                i for i, s in enumerate(signals) if self.TYPE.matches_str(s, q).match
            }
            assert {m.metadata for m in idx.query(q)} == expected, q

    def test_index_loads_old_pickle_state(self) -> None:
        idx = LevenshteinLinearSearch.__new__(LevenshteinLinearSearch)
        idx.__setstate__({"state": [("Hello World", 1)]})
        matches: t.List[IndexMatch[int]] = idx.query("hello world")
        assert [m.metadata for m in matches] == [1]

    @pytest.mark.parametrize(
        "input_val, expected_str",
        [
            ("a", "a"),
            ("a ", "a"),
        ],
    )
    def test_validate_signal_str(self, input_val: str, expected_str: str) -> None:
        validated = self.TYPE.validate_signal_str(input_val)
        assert validated == expected_str, (
            f"Expected {expected_str} for input {input_val}, but got " f"{validated}"
        )
