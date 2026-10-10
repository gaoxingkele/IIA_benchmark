"""Original full MNIST loader/graph audit with all source split identities checked."""
import hashlib
import json
import os
from pathlib import Path

from scripts.flow_matching import audit_adopted_gan_native_v1 as base

original_graph = base.graph_audit


def graph_audit(asset, case, output):
    import numpy as np
    import sys
    sys.path.insert(0, asset['original_directory'])
    import data_utils
    dataset = base.read(case['dataset_configuration'])
    receipt = base.read(Path(dataset['output_directory']) / 'acquisition_and_preprocessing_receipt.json')
    registered = next(item for item in receipt['derived'] if item['split'] == 'train')
    derived = Path(dataset['derived_training_array'])
    if hashlib.sha256(derived.read_bytes()).hexdigest() != registered['sha256']:
        raise ValueError('Complete official training table changed')
    workspace = Path(case['working_directory'])
    workspace.mkdir(parents=True, exist_ok=False)
    (workspace / 'data').mkdir()
    link = workspace / 'data' / 'mnist_train.npy'
    os.link(str(derived), str(link))
    # The source loader only reads this existing NPY. No original CSV/cache is overwritten.
    os.chdir(str(workspace))
    original_loader = data_utils.get_samples_and_labels
    permutations = []

    def observed_loader(settings):
        original_permutation = np.random.permutation

        def permutation(value):
            result = original_permutation(value)
            if isinstance(value, (int, np.integer)) and value == 60000:
                permutations.append(result.copy())
            return result

        np.random.permutation = permutation
        try:
            samples, pdf, labels = original_loader(settings)
        finally:
            np.random.permutation = original_permutation
        if len(permutations) != 1 or not np.array_equal(np.sort(permutations[0]), np.arange(60000)):
            raise ValueError('All 60000 original image indices must occur exactly once')
        table = np.load(str(derived), mmap_mode='r', allow_pickle=False)
        split_sizes = [('train', 36000), ('vali', 12000), ('test', 12000)]
        identities, start = {}, 0
        for split, size in split_sizes:
            indices = permutations[0][start:start + size]
            values = samples[split].reshape(size, 784)
            for offset in range(0, size, 1000):
                selected = indices[offset:offset + 1000]
                if (not np.array_equal(values[offset:offset + 1000], table[selected, 1:])
                        or not np.array_equal(labels[split][offset:offset + 1000], table[selected, 0])):
                    raise ValueError('All native split pixels/labels must equal the original table at exact source indices')
            identities[split] = indices
            start += size
        path = workspace / 'full_native_split_indices.npz'
        np.savez_compressed(str(path), **identities)
        base.write_new(workspace / 'full_native_loader_audit.json',
            dict(full_training_source_images=60000, every_native_split_pixel_and_label_bitwise_verified=True,
                 grouped_disjoint_image_indices=True, training=36000, validation=12000, test=12000,
                 official_test_images_used_for_this_author_split=False, split_indices_path=str(path),
                 split_indices_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), original_loader_unchanged=True))
        return samples, pdf, labels

    data_utils.get_samples_and_labels = observed_loader
    try:
        evidence = original_graph(asset, case, output)
    finally:
        data_utils.get_samples_and_labels = original_loader
    evidence['data']['data_kind'] = 'complete_original_MNIST_training_images_through_unchanged_native_loader'
    evidence['data']['full_native_loader_audit'] = base.read(workspace / 'full_native_loader_audit.json')
    evidence['data']['source_provided_setting'] = 'experiments/settings/mnistfull.txt, 500 epochs, D1/G1, full28x28'
    if hashlib.sha256(derived.read_bytes()).hexdigest() != registered['sha256']:
        raise ValueError('Original derived input changed during native audit')
    return evidence


def main():
    base.graph_audit = graph_audit
    base.main()


if __name__ == '__main__':
    main()
