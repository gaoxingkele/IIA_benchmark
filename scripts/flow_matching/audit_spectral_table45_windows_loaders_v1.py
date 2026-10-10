"""Exercise complete original four-worker loaders and the Windows seed callback."""
import argparse
import json
import os
from pathlib import Path
import sys

from scripts.flow_matching.run_spectral_table45_original_v1 import (
    ROOT, read, verify, seed_worker, row_multiset, sha, write,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args(); config = read(ROOT/args.config)
    for item in config['source_receipts']:
        if sha(ROOT/item['path']) != item['sha256']:
            raise ValueError('Frozen four-worker audit source differs')
    queue = read(ROOT/config['queue']); verify(queue)
    proof = read(ROOT/queue['data_audit_output'])
    if not proof['passed'] or proof['queue_sha256'] != sha(ROOT/config['queue']):
        raise ValueError('Complete original data audit required')
    output = ROOT/config['output']
    if output.exists():
        raise FileExistsError('Preserve previous loader proof')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import torch
    import numpy as np
    torch.set_num_threads(2)
    workspace = Path(proof['workspace']); sys.path.insert(0, str(workspace)); os.chdir(workspace)
    from utils.utils_data import real_data_loading
    np.random.seed(10); torch.manual_seed(10)
    regular = torch.Tensor(real_data_loading('stock', 24))
    spec = proof['records']['pendulum']
    if sha(spec['arrays']) != spec['arrays_sha256']:
        raise ValueError('Original full physics arrays differ')
    with np.load(spec['arrays'], allow_pickle=False) as payload:
        physics = torch.tensor(payload['training'])
    reports = []
    for name, array in [('regular_stock', regular), ('pendulum', physics)]:
        torch.manual_seed(10)
        generator = torch.Generator().manual_seed(10)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(array), batch_size=64,
            shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=generator)
        batches = [batch[0] for batch in loader]
        actual = torch.cat(batches).numpy()
        if len(batches) != (len(array)+63)//64 or row_multiset(actual) != row_multiset(array.numpy()):
            raise ValueError('Complete original four-worker loader lost, duplicated or altered rows')
        reports.append(dict(dataset=name, samples=len(array), shape=list(array.shape), full_batches=len(batches),
            final_batch=len(batches[-1]), original_num_workers=4, full_data_coverage_passed=True,
            windows_spawn_callback_passed=True))
    write(output, dict(full_four_worker_loader_passed=True, diagnostic_only=True, formal_benchmark_result=False,
        queue_sha256=sha(ROOT/config['queue']), data_audit_sha256=sha(ROOT/queue['data_audit_output']),
        cases=reports, boundary='Data loader/Windows compatibility proof only; no model optimization or performance metric.'))


if __name__ == '__main__':
    main()
