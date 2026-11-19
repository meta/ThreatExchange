# Copyright (c) Meta Platforms, Inc. and affiliates.

import io
import pdqhash
import pathlib
import numpy as np
from PIL import Image
import typing as t

PDQOutput = t.Tuple[
    str, int
]  # hexadecimal representation of the Hash vector and a numerical quality value

# Same as the C++ reference (pdq/cpp/io/pdqio.cpp): the Jarosz filters are too
# slow on full-size input, so larger images are first squashed to this size.
DOWNSAMPLE_DIMS = 512


def pdq_from_file(path: pathlib.Path) -> PDQOutput:
    """
    Given a path to a file return the PDQ Hash string in hex.
    Current tested against: jpg
    """
    np_array = _convert_image_to_correct_array_dimension(Image.open(path))
    return _pdq_from_numpy_array(np_array)


def pdq_from_bytes(file_bytes: bytes) -> PDQOutput:
    """
    For the bytestream from an image file, compute PDQ Hash and quality.
    """
    np_array = _convert_image_to_correct_array_dimension(
        Image.open(io.BytesIO(file_bytes))
    )
    return _pdq_from_numpy_array(np_array)


def _pdq_from_numpy_array(array: np.ndarray) -> PDQOutput:
    hash_vector, quality = pdqhash.compute(array)

    bin_str = "".join([str(x) for x in hash_vector])

    # binary to hex using format string
    # '%0*' is for padding up to ceil(num_bits/4),
    # '%X' create a hex representation from the binary string's integer value
    hex_str = "%0*X" % ((len(bin_str) + 3) // 4, int(bin_str, 2))
    hex_str = hex_str.lower()

    return hex_str, quality


def _convert_image_to_correct_array_dimension(image: Image.Image) -> np.ndarray:
    """
    Handle possible image format conversion or resizing.
    """
    if image.mode in ("LA", "I;16", "P", "1"):
        # LA images (luminance with alpha) return 3 dimensional ndarray
        # which is incompatible with pdqhash. P and 1 images would give
        # palette indices and bools instead of pixel values.
        image = image.convert("RGB")

    array = _downsample(np.asarray(image))

    # Convert possible 2D array to 3D array
    # This is more efficient than converting to RGB for mode L images.
    if array.ndim == 2:
        array = np.concatenate([array[..., np.newaxis]] * 3, axis=2)

    return array


def _downsample(array: np.ndarray) -> np.ndarray:
    """
    Nearest-neighbor resize to DOWNSAMPLE_DIMS square, matching CImg's
    resize(512, 512) in the C++ reference pixel for pixel, so hashes agree
    with it exactly for the same decoded pixels.
    """
    rows, cols = array.shape[:2]
    if rows <= DOWNSAMPLE_DIMS and cols <= DOWNSAMPLE_DIMS:
        return array
    # CImg picks source index floor(i * size / DOWNSAMPLE_DIMS) for output index i
    row_idx = np.arange(DOWNSAMPLE_DIMS) * rows // DOWNSAMPLE_DIMS
    col_idx = np.arange(DOWNSAMPLE_DIMS) * cols // DOWNSAMPLE_DIMS
    return np.ascontiguousarray(array[row_idx[:, np.newaxis], col_idx])
