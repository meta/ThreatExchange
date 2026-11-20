#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Wrapper around the raw text signal type.
"""

from dataclasses import dataclass
import typing as t

from rapidfuzz.distance import Levenshtein

from threatexchange import common
from threatexchange.content_type.content_base import ContentType
from threatexchange.content_type.text import TextContent
from threatexchange.signal_type import signal_base
from threatexchange.signal_type import index
from threatexchange.exchanges.impl.fb_threatexchange_signal import (
    HasFbThreatExchangeIndicatorType,
)


@dataclass
class RawTextDistance(index.SignalSimilarityInfo):
    """
    Even though we use simple int thresholds, display as a 0-100% match.

    We re-project it onto the length of the query string.
    """

    distance: float
    query_str_len: int

    @property
    def diff_fraction(self) -> float:
        """0.0 -> 1.0 how different query was from target"""
        return min(self.distance, self.query_str_len) / max(1, self.query_str_len)

    @property
    def match_fraction(self) -> float:
        """0.0 -> 1.0 roughly what percent of the string matched"""
        return 1 - self.diff_fraction

    @classmethod
    def from_levenshtein(cls, needle: str, distance: float) -> "RawTextDistance":
        return cls(distance, len(needle))

    def pretty_str(self) -> str:
        return f"{(self.match_fraction) * 100:.0f}%"


class RawTextSignal(
    signal_base.SimpleSignalType,
    signal_base.MatchesStr,
    HasFbThreatExchangeIndicatorType,
):
    """
    Raw text signal is the same as raw text content: the exact text content.

    Unlike other formats like photos or videos, it is difficult to come
    up with non-reversable hashes of text information which are also effective
    at detecting similar content.
    """

    INDICATOR_TYPE = "TEXT_STRING"
    DEFAULT_PCT_DIFF_THRESHOLD = 5.0

    @classmethod
    def get_content_types(cls) -> t.List[t.Type[ContentType]]:
        return [TextContent]

    @classmethod
    def matches_str(
        cls,
        signal: str,
        haystack: str,
        pct_diff_threshold: float = DEFAULT_PCT_DIFF_THRESHOLD,
    ) -> signal_base.SignalComparisonResult:
        assert 0 < pct_diff_threshold <= 100
        a = common.normalize_string(signal)
        b = common.normalize_string(haystack)
        max_match_distance = cls._max_match_distance(a, pct_diff_threshold)

        ldiff = abs(len(a) - len(b))

        if ldiff > max_match_distance:
            return signal_base.SignalComparisonResult.from_simple_dist(
                ldiff, max_match_distance
            )

        distance = Levenshtein.distance(a, b)

        return signal_base.SignalComparisonResult(
            distance <= max_match_distance,
            RawTextDistance.from_levenshtein(a, distance),
        )

    @staticmethod
    def _max_match_distance(normalized_signal: str, pct_diff_threshold: float) -> float:
        n = len(normalized_signal)
        return n - n * (100 - pct_diff_threshold) / 100

    @classmethod
    def get_index_cls(cls) -> t.Type[index.SignalTypeIndex]:
        return LevenshteinLinearSearch

    @staticmethod
    def get_examples() -> t.List[str]:
        return [
            "The quick brown fox jumps over the lazy dog",
            (
                "We the People of the United States, in Order to form a more "
                "perfect Union, establish Justice, ensure domestic "
                "Tranquility, provide for the common defence, promote the "
                "general Welfare, and secure the Blessings of Liberty to "
                "ourselves and our Posterity, do ordain and establish this "
                "Constitution for the United States of America."
            ),
            "bball now?",
        ]


class LevenshteinLinearSearch(signal_base.TrivialLinearSearchMatchIndex):
    """Linear search that normalizes each stored signal once instead of per query."""

    _SIGNAL_TYPE = RawTextSignal

    def __init__(self) -> None:
        super().__init__()
        self._normalized: t.List[str] = []

    def add(self, signal_str: str, entry: index.T) -> None:
        super().add(signal_str, entry)
        self._normalized.append(common.normalize_string(signal_str))

    def query(self, query_hash: str) -> t.List[index.IndexMatch[index.T]]:
        # Same result as matches_str on each entry, without its per-entry overhead
        b = common.normalize_string(query_hash)
        len_b = len(b)
        pct = RawTextSignal.DEFAULT_PCT_DIFF_THRESHOLD
        ret = []
        for a, (_, payload) in zip(self._normalized, self.state):
            max_dist = RawTextSignal._max_match_distance(a, pct)
            if abs(len(a) - len_b) > max_dist:
                continue
            distance = Levenshtein.distance(a, b, score_cutoff=int(max_dist))
            if distance <= max_dist:
                ret.append(
                    index.IndexMatch(
                        RawTextDistance.from_levenshtein(a, distance), payload
                    )
                )
        return ret

    def __getstate__(self) -> t.Dict[str, t.Any]:
        # Keep the pickled format identical to older versions
        state = self.__dict__.copy()
        del state["_normalized"]
        return state

    def __setstate__(self, state: t.Dict[str, t.Any]) -> None:
        self.__dict__.update(state)
        self._normalized = [common.normalize_string(s) for s, _ in self.state]
