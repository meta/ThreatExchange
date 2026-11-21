# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest
import binascii
import pickle
import typing as t
import numpy

from threatexchange.signal_type.pdq.pdq_faiss_matcher import (
    PDQFlatHashIndex,
    PDQMultiHashIndex,
)

test_hashes = [
    "0000000000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000ffff",
    "0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f",
    "f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0",
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
]

MAX_UNSIGNED_INT64 = numpy.iinfo(numpy.int64).max


class MixinTests:
    class PDQHashIndexCommonTests(unittest.TestCase):
        index = None

        def assertEqualPDQHashSearchResults(self, result, expected):
            self.assertEqual(
                len(result), len(expected), "search results not of expected length"
            )
            for r, e in zip(result, expected):
                self.assertCountEqual(r, e)

        def test_search_index_for_exact_matches(self):
            query = test_hashes[:1]
            result = self.index.search(query, 0)
            self.assertEqualPDQHashSearchResults(result, [[q] for q in query])

        def test_search_index_for_near_matches(self):
            query = test_hashes[:1]
            result = self.index.search(query, 16)
            self.assertEqualPDQHashSearchResults(result, [test_hashes[:2]])

        def test_search_index_for_far_match(self):
            query = test_hashes[:1]
            result = self.index.search(query, 128)
            self.assertEqualPDQHashSearchResults(result, [test_hashes[:-1]])

        def test_search_index_multiple_queries(self):
            query = test_hashes
            result = self.index.search(query, 0)
            self.assertEqualPDQHashSearchResults(result, [[h] for h in test_hashes])

        def test_search_index_with_no_match(self):
            query = ["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]
            result = self.index.search(query, 0)
            self.assertEqualPDQHashSearchResults(result, [[]])

        def test_search_index_with_multiple_no_matches(self):
            query = [
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ]
            result = self.index.search(query, 0)
            self.assertEqualPDQHashSearchResults(result, [[], []])

        def test_search_index_with_multiple_mixed_results(self):
            query = [
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                test_hashes[-1],
            ]
            result = self.index.search(query, 0)
            self.assertEqualPDQHashSearchResults(result, [[], [], [test_hashes[-1]]])

        def test_supports_pickling(self):
            pickled_data = pickle.dumps(self.index)
            assert (
                pickled_data != None
            ), "index does not support pickling to a data stream"

            reconstructed_index = pickle.loads(pickled_data)
            assert (
                reconstructed_index != None
            ), "index does not support unpickling from data stream"
            assert (
                reconstructed_index.faiss_index != self.index.faiss_index
            ), "unpickling should create it's own faiss index in memory"

            query = [test_hashes[0]]
            result = reconstructed_index.search(query, 0)
            self.assertEqualPDQHashSearchResults(result, [[test_hashes[0]]])


class TestPDQFlatHashIndex(MixinTests.PDQHashIndexCommonTests, unittest.TestCase):
    def setUp(self):
        self.index = PDQFlatHashIndex()
        self.index.add(test_hashes, range(0, len(test_hashes)))

    def test_create_faiss_index_from_hashes(self):
        assert type(self.index) is PDQFlatHashIndex
        hashes_as_bytes = [binascii.unhexlify(h) for h in test_hashes]

        assert self.index.faiss_index is not None
        assert self.index.faiss_index.ntotal == len(test_hashes)

    def test_hash_at(self):
        assert test_hashes[2] == self.index.hash_at(2)

    def test_search_index_return_ids(self):
        query = test_hashes[:2]
        results = self.index.search(query, 16, return_as_ids=True)
        self.assertEqualPDQHashSearchResults(results, [[0, 1], [0, 1]])


class TestPDQFlatHashIndexWithCustomIds(
    MixinTests.PDQHashIndexCommonTests, unittest.TestCase
):
    custom_ids = [MAX_UNSIGNED_INT64 - i for i in range(len(test_hashes))]

    def setUp(self):
        self.index = PDQFlatHashIndex()
        self.index.add(test_hashes, self.custom_ids)

    def test_create_faiss_index_from_hashes(self):
        assert type(self.index) is PDQFlatHashIndex
        hashes_as_bytes = [binascii.unhexlify(h) for h in test_hashes]

        assert self.index.faiss_index is not None
        assert self.index.faiss_index.ntotal == len(test_hashes)

    def test_hash_at(self):
        assert test_hashes[2] == self.index.hash_at(self.custom_ids[2])

    def test_search_index_return_ids(self):
        query = test_hashes[:2]
        results = self.index.search(query, 16, return_as_ids=True)
        self.assertEqualPDQHashSearchResults(
            results, [self.custom_ids[:2], self.custom_ids[:2]]
        )


class TestPDQMultiHashIndex(MixinTests.PDQHashIndexCommonTests, unittest.TestCase):
    def setUp(self):
        self.index = PDQMultiHashIndex()
        self.index.add(test_hashes, range(0, len(test_hashes)))

    def test_create_faiss_index_from_hashes(self):
        assert type(self.index) is PDQMultiHashIndex
        hashes_as_bytes = [binascii.unhexlify(h) for h in test_hashes]

        assert self.index.faiss_index is not None
        assert self.index.faiss_index.ntotal == len(test_hashes)

    def test_hash_at(self):
        assert test_hashes[2] == self.index.hash_at(2)

    def test_search_index_return_ids(self):
        query = test_hashes[:2]
        results = self.index.search(query, 16, return_as_ids=True)
        self.assertEqualPDQHashSearchResults(results, [[0, 1], [0, 1]])


class TestPDQMultiHashIndexWithCustomIds(
    MixinTests.PDQHashIndexCommonTests, unittest.TestCase
):
    custom_ids = [MAX_UNSIGNED_INT64 - i for i in range(len(test_hashes))]

    def setUp(self):
        self.index = PDQMultiHashIndex()
        self.index.add(test_hashes, custom_ids=self.custom_ids)

    def test_create_faiss_index_from_hashes(self):
        assert type(self.index) is PDQMultiHashIndex
        hashes_as_bytes = [binascii.unhexlify(h) for h in test_hashes]

        assert self.index.faiss_index is not None
        assert self.index.faiss_index.ntotal == len(test_hashes)

    def test_hash_at(self):
        assert test_hashes[2] == self.index.hash_at(self.custom_ids[2])

    def test_search_index_return_ids(self):
        query = test_hashes[:2]
        results = self.index.search(query, 16, return_as_ids=True)
        self.assertEqualPDQHashSearchResults(
            results, [self.custom_ids[:2], self.custom_ids[:2]]
        )


VALID_HASH = test_hashes[1]
INVALID_HASH_CASES = {
    "too_short": "ab" * 16,
    "too_long": "ab" * 33,
    "odd_length": "a" * 63,
    "empty": "",
    "non_hex_chars": "g" * 64,
    "empty_bytes": b"",
    "short_bytes": b"ab" * 16,
}


class MixinValidationTests:
    class PDQHashValidationTests(unittest.TestCase):
        index_class: t.Any = None

        def setUp(self):
            self.index = self.index_class()
            self.index.add(test_hashes, range(len(test_hashes)))

        def assertIndexUnchanged(self):
            self.assertEqual(self.index.faiss_index.ntotal, len(test_hashes))
            self.assertEqual(self.index.search([test_hashes[0]], 0), [[test_hashes[0]]])

        def test_valid_64_char_hash_and_bytes_are_accepted(self):
            index = self.index_class()
            index.add([VALID_HASH, VALID_HASH.encode()], [1, 2])
            self.assertEqual(index.faiss_index.ntotal, 2)
            self.assertCountEqual(
                index.search([VALID_HASH], 0, return_as_ids=True)[0], [1, 2]
            )

        def test_add_rejects_each_invalid_hash(self):
            for name, bad in INVALID_HASH_CASES.items():
                with self.subTest(name):
                    with self.assertRaisesRegex(ValueError, "position 1"):
                        self.index.add([VALID_HASH, bad], [100, 101])
                    self.assertIndexUnchanged()

        def test_add_mixed_valid_and_invalid_leaves_index_unchanged(self):
            batch = [VALID_HASH, VALID_HASH, "ab" * 16, VALID_HASH]
            with self.assertRaisesRegex(ValueError, "position 2"):
                self.index.add(batch, range(100, 104))
            self.assertIndexUnchanged()

        def test_add_uniformly_short_hashes_rejected(self):
            with self.assertRaisesRegex(ValueError, "position 0.*16 bytes"):
                self.index.add(["ab" * 16, "cd" * 16], [100, 101])
            self.assertIndexUnchanged()

        def test_add_rejects_mismatched_id_count(self):
            with self.assertRaises(ValueError):
                self.index.add([VALID_HASH, VALID_HASH], [100])
            self.assertIndexUnchanged()

        def test_search_rejects_invalid_query(self):
            for name, bad in INVALID_HASH_CASES.items():
                with self.subTest(name):
                    with self.assertRaisesRegex(ValueError, "query hash at position 1"):
                        self.index.search([VALID_HASH, bad], 0)
                    with self.assertRaisesRegex(ValueError, "query hash at position 1"):
                        self.index.search_with_distance_in_result([VALID_HASH, bad], 0)

        def test_valid_matching_unchanged(self):
            self.assertEqual(self.index.search([test_hashes[0]], 16), [test_hashes[:2]])
            result = self.index.search_with_distance_in_result([test_hashes[0]], 16)
            self.assertCountEqual(
                [(i, h, int(d)) for i, h, d in result[test_hashes[0]]],
                [(0, test_hashes[0], 0), (1, test_hashes[1], 16)],
            )


class TestPDQFlatHashValidation(MixinValidationTests.PDQHashValidationTests):
    index_class = PDQFlatHashIndex


class TestPDQMultiHashValidation(MixinValidationTests.PDQHashValidationTests):
    index_class = PDQMultiHashIndex


if __name__ == "__main__":
    unittest.main()
