"""Explicit paired local probe draws for the paper-derived image scorer."""
import hashlib
import torch

from .fm_supplemental import flow_mismatch


def probe_seed(training_seed, dataset, native_image_index, path_index):
    if native_image_index < 0 or path_index < 0:
        raise ValueError('Native image and path indices must be nonnegative')
    key = '%d/%s/%d/%d' % (training_seed, dataset, native_image_index, path_index)
    return int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:8], 'little') & ((1 << 63) - 1)


def paired_probes(image, *, training_seed, dataset, native_image_index, paths):
    if image.shape[0] != 1 or image.ndim != 4 or paths < 1:
        raise ValueError('One complete original image and positive path count required')
    seeds, probes = [], []
    for index in range(paths):
        seed = probe_seed(training_seed, dataset, native_image_index, index)
        generator = torch.Generator(device=image.device).manual_seed(seed)
        probes.append(torch.randn(image.shape, dtype=image.dtype, device=image.device, generator=generator))
        seeds.append(seed)
    return torch.stack(probes), seeds


@torch.inference_mode()
def score_original_image(model, image, *, labels, parameters, training_seed,
                         dataset, native_image_index):
    if model.training:
        raise ValueError('Complete evaluation requires model.eval()')
    probes, seeds = paired_probes(image, training_seed=training_seed, dataset=dataset,
                                  native_image_index=native_image_index, paths=parameters['paths'])
    heatmap, score = flow_mismatch(model, image, labels=labels, seeds=probes, **parameters)
    return heatmap, score, seeds
