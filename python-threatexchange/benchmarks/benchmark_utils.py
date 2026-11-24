# Copyright (c) Meta Platforms, Inc. and affiliates.

import binascii
import numpy

from threatexchange.signal_type.pdq.pdq_utils import BITS_IN_PDQ

def generate_random_hash(rng):
    """
    returns a random 256 bit PDQ hash as a hexstring of 64 characters
    """
    hash_bytes = rng.bytes(BITS_IN_PDQ // 8)
    return binascii.hexlify(hash_bytes).decode()


def generate_random_distance_mask(rng, hamming_distance):
    """
    returns a random numpy array of uint8s that can be used as bitwise mask
    to generate a hash with the given hamming distance
    """
    ones = numpy.ones(hamming_distance, dtype=numpy.uint8)
    bitmask = numpy.pad(
        ones, (0, BITS_IN_PDQ - hamming_distance), "constant", constant_values=0
    )
    return numpy.packbits(rng.permutation(bitmask))


def generate_random_hash_with_hamming_distance(rng, original_hash, desired_hamming_distance):
    """
    returns a random 256 bit PDQ hash as a hexstring of 64 characters that is the given
    hamming distance from the provided original hash
    """
    original_hash_bytes = numpy.frombuffer(
        binascii.unhexlify(original_hash), dtype=numpy.uint8
    )
    mask = generate_random_distance_mask(rng, desired_hamming_distance)
    new_hash_bytes = numpy.bitwise_xor(original_hash_bytes, mask).tobytes()
    return binascii.hexlify(new_hash_bytes).decode()
