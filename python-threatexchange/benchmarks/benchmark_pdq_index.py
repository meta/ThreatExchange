# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import pickle
import time
import typing as t

import faiss
import numpy

from threatexchange.signal_type.index import SignalTypeIndex
from threatexchange.signal_type.pdq.pdq_index import PDQFlatIndex, PDQIndex
from threatexchange.signal_type.pdq.pdq_index2 import PDQIndex2
from threatexchange.signal_type.pdq.pdq_utils import BITS_IN_PDQ

from benchmarks.benchmark_utils import (
    generate_random_hash,
    generate_random_hash_with_hamming_distance,
)

# Matches PDQMultiHashIndex's default configuration
MULTI_HASH_NHASH = 16

parser = argparse.ArgumentParser(
    description="Run basic benchmarks comparing PDQIndex2 to PDQIndex using faiss",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--faiss-threads",
    type=int,
    default=1,
    help="number of threads for faiss to use while searching",
)
parser.add_argument(
    "--dataset-size",
    type=int,
    default=10000,
    help="number of hashes to generate for the dataset to search against",
)
parser.add_argument(
    "--num-queries",
    type=int,
    default=1000,
    help="number of queries to generate for each search",
)
parser.add_argument("--seed", type=int, help="seed for random number generator")

args = parser.parse_args()

######
# Print Benchmark Settings
######

print("Benchmark: PDQIndex2 & PDQIndex")
print("")
print("Options:")
for arg in vars(args):
    print("\t", arg, ": ", getattr(args, arg))
print("")

######
# Set up environment and helpers
######

faiss.omp_set_num_threads(args.faiss_threads)
seed = args.seed if args.seed is not None else time.time_ns()
rng = numpy.random.default_rng(seed)
if args.seed is None:
    print("using random seed of ", seed)
    print("use --seed ", seed, " to rerun with same random values")
    print("")

######
# Index configurations
######

IndexFactory = t.Callable[[], SignalTypeIndex[int]]


def make_float_flat_index2(threshold: int) -> IndexFactory:
    return lambda: PDQIndex2(faiss.IndexFlatL2(BITS_IN_PDQ), threshold=threshold)


# Grouped by threshold, derived from PDQIndex class values
GROUPS: t.List[t.Tuple[int, t.List[t.Tuple[str, IndexFactory]]]] = [
    (
        PDQIndex.get_match_threshold(),
        [
            ("PDQIndex (IndexBinaryMultiHash)", PDQIndex),
            (
                "PDQIndex2 (IndexFlatL2)",
                make_float_flat_index2(PDQIndex.get_match_threshold()),
            ),
        ],
    ),
    (
        PDQFlatIndex.get_match_threshold(),
        [
            ("PDQFlatIndex (IndexBinaryFlat)", PDQFlatIndex),
            (
                "PDQIndex2 (IndexFlatL2)",
                make_float_flat_index2(PDQFlatIndex.get_match_threshold()),
            ),
        ],
    ),
]

######
# Generate Random Dataset
######

dataset = [generate_random_hash(rng) for _ in range(args.dataset_size)]
# Unique entry per hash, so a found target is identified by its position
entries = [(h, i) for i, h in enumerate(dataset)]

######
# Build, then search, each index for each group
######

for threshold, group in GROUPS:
    print("Benchmarks for threshold: ", threshold)

    # Create queries with hamming distance of threshold compared to their search targets
    target_ids = rng.integers(0, args.dataset_size, size=args.num_queries).tolist()
    queries = [
        generate_random_hash_with_hamming_distance(rng, dataset[target_id], threshold)
        for target_id in target_ids
    ]

    for name, factory in group:
        start = time.time()
        index = factory()
        index.add_all(entries)
        build_time = time.time() - start

        start = time.time()
        serialized = pickle.dumps(index)
        serialize_time = time.time() - start
        size_kb = len(serialized) // 1024
        del serialized

        start = time.time()
        results = [index.query(q) for q in queries]
        search_time = time.time() - start

        found = sum(
            1
            for target_id, result in zip(target_ids, results)
            if any(match.metadata == target_id for match in result)
        )

        print(f"\t{name}")
        print(f"\t\tTime to build (s): {build_time}")
        print(f"\t\tTime to serialize (s): {serialize_time}")
        print(f"\t\tApproximate size: {size_kb:,d}KB")
        print(f"\t\tTotal time to search (s): {search_time}")
        print(f"\t\tMean time per query (ms): {search_time / len(queries) * 1000}")
        print(f"\t\tPercent of targets found: {found / len(queries) * 100}")
        del index, results

    print("")
