"""Acquire complete checksum-pinned MNIST; preserve raw and original preprocessing."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import struct
import subprocess

import numpy as np


def parse_idx(payload, expected_shape):
    if len(payload) < 4 or payload[:2] != b'\0\0' or payload[2] != 8:
        raise ValueError('Expected uint8 IDX header')
    dimensions = payload[3]
    if dimensions not in (1, 3) or len(payload) < 4 + 4 * dimensions:
        raise ValueError('Expected complete label/image IDX dimensions')
    shape = list(struct.unpack('>' + 'I' * dimensions, payload[4:4 + 4 * dimensions]))
    if shape != expected_shape or len(payload) != 4 + 4 * dimensions + int(np.prod(shape)):
        raise ValueError('Every registered original IDX item and pixel is required')
    values = np.frombuffer(payload, dtype=np.uint8, offset=4 + 4 * dimensions).reshape(shape)
    if dimensions == 1 and np.any(values > 9):
        raise ValueError('Original MNIST labels must be 0 through 9')
    return values


def original_scaled_table(images, labels):
    if len(images) != len(labels):
        raise ValueError('Every image requires its original ordered label')
    table = np.empty((len(images), 785), dtype=np.float64)
    table[:, 0] = labels
    table[:, 1:] = images.reshape(len(images), 784)
    table[:, 1:] /= 255
    table[:, 1:] = 2 * table[:, 1:] - 1
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    raw = Path(config['raw_directory'])
    raw.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(Path(config['official_manifest']['local_path']).read_bytes()).hexdigest() != config['official_manifest']['sha256']:
        raise ValueError('Pinned official dataset manifest changed')

    def acquire(asset):
        path = raw / asset['filename']
        if not path.exists():
            subprocess.run([config['aria2c'], '--all-proxy=' + config['proxy'], '--allow-overwrite=false',
                            '--auto-file-renaming=false', '--continue=true', '--max-tries=2', '--retry-wait=2',
                            '--checksum=md5=' + asset['md5'], '--dir=' + str(raw), '--out=' + asset['filename'], asset['url']],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        payload = path.read_bytes()
        if hashlib.md5(payload).hexdigest() != asset['md5']:
            raise ValueError('Original official MNIST checksum differs')
        return dict(path=str(path), bytes=len(payload), md5=asset['md5'], sha256=hashlib.sha256(payload).hexdigest())

    with ThreadPoolExecutor(max_workers=4) as pool:
        assets = list(pool.map(acquire, config['assets']))
    parsed = {}
    filenames = {'train_images': 'train-images-idx3-ubyte.gz', 'train_labels': 'train-labels-idx1-ubyte.gz',
                 'test_images': 't10k-images-idx3-ubyte.gz', 'test_labels': 't10k-labels-idx1-ubyte.gz'}
    for key, filename in filenames.items():
        parsed[key] = parse_idx(gzip.decompress((raw / filename).read_bytes()), config['full_shapes'][key])
    derived = []
    for split in ('train', 'test'):
        path = Path(config['derived_training_array' if split == 'train' else 'derived_test_array'])
        path.parent.mkdir(parents=True, exist_ok=True)
        table = original_scaled_table(parsed[split + '_images'], parsed[split + '_labels'])
        expected = config['derived_training_shape' if split == 'train' else 'derived_test_shape']
        if list(table.shape) != expected:
            raise ValueError('Complete original table required')
        with path.open('xb') as handle:
            np.save(handle, table, allow_pickle=False)
        actual = np.load(path, mmap_mode='r', allow_pickle=False)
        if not np.array_equal(actual, table):
            raise ValueError('Every derived value must equal the original formula exactly')
        if not np.array_equal(np.rint((actual[:, 1:] + 1) / 2 * 255).astype(np.uint8),
                              parsed[split + '_images'].reshape(len(actual), 784)):
            raise ValueError('All original pixels must roundtrip')
        if not np.array_equal(actual[:, 0], parsed[split + '_labels']):
            raise ValueError('All original labels/order must be preserved')
        derived.append(dict(split=split, path=str(path), bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), full_shape=list(actual.shape),
                            dtype=str(actual.dtype), every_pixel_and_label_independently_checked=True,
                            label_counts={str(i): int(np.sum(actual[:, 0] == i)) for i in range(10)}))
    for asset, record in zip(config['assets'], assets):
        if hashlib.md5(Path(record['path']).read_bytes()).hexdigest() != asset['md5']:
            raise ValueError('Raw source changed during preprocessing')
    output = Path(config['output_directory'])
    output.mkdir(parents=True, exist_ok=True)
    receipt = dict(captured_utc=datetime.now(timezone.utc).isoformat(), configuration=args.config,
                   config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(), raw_files=assets, derived=derived,
                   original_normalization_preserved=True, full_official_training_images=60000, full_official_test_images=10000,
                   raw_sources_unchanged=True, training_started=False, benchmark_performance=False,
                   historical_csv_byte_identity_certified=False, all_original_experiments_complete=False)
    with (output / 'acquisition_and_preprocessing_receipt.json').open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(receipt, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(raw_files_verified=4, full_train_images=60000, full_test_images=10000,
                         original_preprocessing_exact=True, benchmark_performance=False)))


if __name__ == '__main__':
    main()
