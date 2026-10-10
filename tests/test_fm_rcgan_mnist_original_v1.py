import struct

import numpy as np
import pytest

from scripts.flow_matching.prepare_rcgan_mnist_original_v1 import original_scaled_table, parse_idx


def test_idx_requires_complete_ordered_pixels_and_dimensions():
    pixels = bytes(range(8))
    payload = b'\0\0\x08\x03' + struct.pack('>III', 2, 2, 2) + pixels
    assert np.array_equal(parse_idx(payload, [2, 2, 2]).reshape(-1), list(range(8)))
    for wrong in (payload[:-1], payload + b'\0', payload[:2] + b'\x09' + payload[3:]):
        with pytest.raises(ValueError):
            parse_idx(wrong, [2, 2, 2])
    with pytest.raises(ValueError):
        parse_idx(payload, [1, 2, 2])


def test_idx_rejects_non_mnist_labels():
    payload = b'\0\0\x08\x01' + struct.pack('>I', 2) + b'\0\x0a'
    with pytest.raises(ValueError, match='labels'):
        parse_idx(payload, [2])


def test_preprocessing_matches_original_csv_arithmetic_and_keeps_label_order():
    images = np.arange(2 * 784, dtype=np.uint16).astype(np.uint8).reshape(2, 28, 28)
    labels = np.array([8, 3], dtype=np.uint8)
    actual = original_scaled_table(images, labels)
    expected = images.astype(np.float64).reshape(2, 784)
    expected /= 255
    expected = 2 * expected - 1
    assert actual.dtype == np.float64
    assert np.array_equal(actual[:, 0], labels)
    assert np.array_equal(actual[:, 1:], expected)
    assert np.array_equal(np.rint((actual[:, 1:] + 1) / 2 * 255).astype(np.uint8), images.reshape(2, 784))
