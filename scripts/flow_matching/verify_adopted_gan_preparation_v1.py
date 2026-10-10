"""Independent full asset, native proof, MIDI coverage and MNIST pixel verification."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'projects/flow_matching_research/results/2026-10-11/adopted_gan_original_sources_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    registry = read(ROOT / 'configs/acquisition/fm_adopted_gan_original_sources.v1.json')
    receipt = read(REPORT / 'acquisition_receipt.json')
    if digest(ROOT / receipt['configuration']) != receipt['configuration_sha256']:
        raise ValueError('Frozen acquired source inventory differs')
    total_files, total_bytes = 0, 0
    for asset in registry['assets']:
        original = Path(asset['original_directory'])
        actual_paths = {file.relative_to(original).as_posix() for file in original.rglob('*') if file.is_file()}
        expected_paths = {entry['path'] for entry in asset['expected_git_entries']}
        if actual_paths != expected_paths:
            raise ValueError('Full author source tree membership changed')
        saved = next(record for record in receipt['original_repositories'] if record['id'] == asset['id'])
        file_receipts = {record['path']: record for record in saved['file_receipts']}
        for entry in asset['expected_git_entries']:
            payload = (original / entry['path']).read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(payload)).encode('ascii') + b'\0' + payload).hexdigest()
            if (len(payload) != entry['size'] or blob != entry['sha']
                    or hashlib.sha256(payload).hexdigest() != file_receipts[entry['path']]['sha256']):
                raise ValueError('Every original author Git blob and published file receipt must match')
            total_files += 1
            total_bytes += len(payload)
    if total_files != 3534:
        raise ValueError('Both complete pinned author source trees required')

    native_root = ROOT / 'projects/flow_matching_research/results/2026-10-11/adopted_gan_native_preparation_v1'
    specifications = [('rcgan_full_source_default_gp_graph', 83402, [28, 30, 1]),
                      ('crnngan_paper_width350_tones1_full_graph', 4924855, [20, 100, 4]),
                      ('crnngan_paper_width350_tones3_full_graph', 4947961, [20, 100, 10])]
    for identifier, parameters, shape in specifications:
        proof = read(native_root / (identifier + '.json'))
        execution = read(native_root / (identifier + '.execution_receipt.json'))
        evidence = proof['evidence']
        if (not proof['passed'] or not proof['source_and_raw_files_unchanged'] or execution['exit_code'] != 0
                or execution['resource_guard_aborted'] or not evidence['full_native_initialization_and_forward']
                or evidence['optimizer_updates'] != 0 or evidence['model_parameters'] != parameters
                or evidence['generated_shape'] != shape or evidence['tensorflow'] != '1.15.0'
                or evidence['numpy'] != '1.18.5' or evidence['benchmark_performance']):
            raise ValueError('Exact full native preparation proof required, never benchmark training')

    crnn = next(asset for asset in registry['assets'] if asset['id'] == 'crnngan')
    midi_paths = {entry['path']: entry['size'] for entry in crnn['expected_git_entries']
                  if entry['path'].lower().endswith('.mid')}
    midi_proof = read(ROOT / 'projects/flow_matching_research/results/2026-10-11/adopted_gan_native_preparation_v2/crnngan_all_released_midi.json')
    records = midi_proof['evidence']['records']
    if (len(records) != 3451 or {record['path']: record['bytes'] for record in records} != midi_paths
            or sum(record['parsed'] for record in records) != 3435
            or sum(not record['parsed'] for record in records) != 16
            or sum(record.get('note_events', 0) for record in records) != 10661408
            or not midi_proof['source_and_raw_files_unchanged']):
        raise ValueError('Every released MIDI file outcome, including the sixteen author-file failures, required')
    quality = read(REPORT / 'crnngan_genre_resolved_data_audit.json')
    classical = [record for record in records if record['path'].startswith('data/classical/')]
    if (len(classical) != 3029 or sum(record['parsed'] for record in classical) != 3014
            or quality['classical_file_count_gap'] != 668 or quality['classical_composer_count_gap'] != 33
            or quality['all_released_files_usable'] or quality['historical_paper_dataset_identity_certified']):
        raise ValueError('Keep the historical classical corpus gap and actual raw failures explicit')

    mnist_config = read(ROOT / 'configs/acquisition/fm_rcgan_mnist_original.v1.json')
    mnist_receipt = read(ROOT / mnist_config['output_directory'] / 'acquisition_and_preprocessing_receipt.json')
    raw = Path(mnist_config['raw_directory'])
    for asset in mnist_config['assets']:
        if hashlib.md5((raw / asset['filename']).read_bytes()).hexdigest() != asset['md5']:
            raise ValueError('Full official raw MNIST checksum differs')
    for split, stem, count in [('train', 'train', 60000), ('test', 't10k', 10000)]:
        image_bytes = gzip.decompress((raw / (stem + '-images-idx3-ubyte.gz')).read_bytes())
        label_bytes = gzip.decompress((raw / (stem + '-labels-idx1-ubyte.gz')).read_bytes())
        if struct.unpack('>IIII', image_bytes[:16]) != (2051, count, 28, 28):
            raise ValueError('All official MNIST images/header required')
        if struct.unpack('>II', label_bytes[:8]) != (2049, count):
            raise ValueError('All official MNIST labels/header required')
        images = np.frombuffer(image_bytes, dtype=np.uint8, offset=16).reshape(count, 784)
        labels = np.frombuffer(label_bytes, dtype=np.uint8, offset=8)
        derived = next(record for record in mnist_receipt['derived'] if record['split'] == split)
        if digest(derived['path']) != derived['sha256']:
            raise ValueError('Full derived MNIST file changed')
        actual = np.load(derived['path'], mmap_mode='r', allow_pickle=False)
        if list(actual.shape) != [count, 785] or actual.dtype != np.float64:
            raise ValueError('All original normalized rows/features required')
        for offset in range(0, count, 1000):
            expected = images[offset:offset + 1000].astype(np.float64)
            expected /= 255
            expected = 2 * expected - 1
            if (not np.array_equal(actual[offset:offset + 1000, 1:], expected)
                    or not np.array_equal(actual[offset:offset + 1000, 0], labels[offset:offset + 1000])):
                raise ValueError('Every complete normalized original pixel and label must match')

    mnist_native = ROOT / 'projects/flow_matching_research/results/2026-10-11/rcgan_mnist_native_preparation_v1'
    reconciliation = read(mnist_native / 'native_report_reconciliation.v1.json')
    original_path = reconciliation['original_native_report_path']
    reconciled_path = ROOT / reconciliation['reconciled_read_only_copy']
    if Path(original_path).read_bytes() != reconciled_path.read_bytes() or digest(reconciled_path) != reconciliation['sha256']:
        raise ValueError('Preserve exact misplaced original native proof bytes')
    proof = read(reconciled_path)
    evidence = proof['evidence']
    loader = evidence['data']['full_native_loader_audit']
    if (not proof['passed'] or evidence['model_parameters'] != 96929 or evidence['generated_shape'] != [28, 28, 28]
            or evidence['optimizer_updates'] != 0 or not loader['every_native_split_pixel_and_label_bitwise_verified']):
        raise ValueError('Complete actual source loader and original MNIST graph required')
    if digest(loader['split_indices_path']) != loader['split_indices_sha256']:
        raise ValueError('Complete original source split indices changed')
    with np.load(loader['split_indices_path'], allow_pickle=False) as archive:
        if any(archive[key].shape != (count,) for key, count in [('train', 36000), ('vali', 12000), ('test', 12000)]):
            raise ValueError('All original split identities required')
        if not np.array_equal(np.sort(np.concatenate([archive[key] for key in ['train', 'vali', 'test']])), np.arange(60000)):
            raise ValueError('Full disjoint image coverage required')
    report = dict(captured_utc=datetime.now(timezone.utc).isoformat(), passed=True,
                  all_original_git_files_reverified=total_files, original_file_bytes_reverified=total_bytes,
                  full_native_graphs_verified=4, original_algorithm_source_edits=0,
                  all_released_midi_files_visited=3451, parsed_midi_files=3435, preserved_rejected_midi_files=16,
                  full_classical_released_files=3029, paper_classical_count_gap=668,
                  full_original_mnist_train_images=60000, full_original_mnist_test_images=10000,
                  all_mnist_pixels_and_labels_reverified=True, full_native_mnist_split_identities_reverified=True,
                  native_report_cwd_failure_preserved_and_exact_proof_reconciled=True,
                  optimizer_updates=0, benchmark_performance_rows_added=0,
                  author_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False)
    with (REPORT / 'independent_full_preparation_validation.json').open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
