"""CPU-only full architecture/environment capacity record, no performance test."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'src'))
    import torch
    from iia_benchmark.models.fm_supplemental_author import FlowMismatchUNet
    torch.set_num_threads(2)
    values = []
    for name in ('mvtec', 'visa'):
        config = read(ROOT / ('configs/models/fm_flow_mismatching_' + name + '_original.v1.json'))
        for variant in ('conditional', 'pooled'):
            parameters = dict(config['parameters'])
            if variant == 'pooled':
                parameters['num_classes'] = None
            model = FlowMismatchUNet(**parameters, asset_root=ROOT)
            count = sum(p.numel() for p in model.parameters())
            buffers = sum(b.numel() * b.element_size() for b in model.buffers())
            # FP32 parameters plus first/second/AMSGrad moments, full resume state.
            estimate = count * 16 + buffers + 8 * 1024 ** 2
            values.append(dict(dataset=name, variant=variant, parameter_count=count,
                parameter_dtype=str(next(model.parameters()).dtype), buffer_bytes=buffers,
                conservative_full_checkpoint_bytes=estimate,
                architecture_parameters=parameters, forward_training_executed=False))
            del model
    total = sum(v['conservative_full_checkpoint_bytes'] * 5 * 22 for v in values)
    target = ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(dict(captured_utc=datetime.now(timezone.utc).isoformat(),
            python_executable=str(Path(sys.executable).resolve()), python_sha256=sha(sys.executable),
            python_version=sys.version,
            packages={p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'psutil')},
            cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
            gpu_memory_capacity_validated=False, full_architectures=values,
            estimated_twenty_jobs_periodic_and_rolling_checkpoint_bytes=total,
            required_free_disk_bytes=total + 32 * 1024 ** 3,
            benchmark_performance=False,
            boundary='CPU full model instantiation and storage estimate; no image forward, backward, epoch, CUDA capacity or accuracy claim.'), indent=2) + '\n')
    print(json.dumps(dict(output=str(target), parameter_counts=[v['parameter_count'] for v in values],
                         checkpoint_storage_estimate_bytes=total)))


if __name__ == '__main__':
    main()
