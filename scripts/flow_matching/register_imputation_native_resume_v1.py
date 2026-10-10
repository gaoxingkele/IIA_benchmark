"""Bind every existing GRIN/transfer job, original phase gate and author source."""
from pathlib import Path
import argparse
import json

import yaml

from scripts.flow_matching.run_light_controller import ROOT, digest


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=['grin_original', 'transfer'])
    args = parser.parse_args()
    common = {
        'scripts/flow_matching/run_imputation_native_resume_queue_v1.py',
        'scripts/flow_matching/register_imputation_native_resume_v1.py',
        'scripts/flow_matching/run_queue.py',
        'scripts/flow_matching/run_tsad_native_gpu_resource_queue_v1.py',
        'scripts/flow_matching/run_spectral_original_queue.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/run_light_controller.py',
        'scripts/flow_matching/run_resource_gated_light_controller_v1.py',
        'scripts/flow_matching/run_grasp_protocol_queue.py',
        'scripts/flow_matching/run_tsad_queue.py',
        'scripts/flow_matching/heavy_resources.py',
        'scripts/flow_matching/heavy_scheduler_v2.py',
        'scripts/flow_matching/phase_gate.py',
        'scripts/flow_matching/metrics.py',
        'tests/test_fm_imputation_native_resume_v1.py',
    }
    settings = dict(minimum_available_memory_bytes=18 * 2**30,
        minimum_commit_headroom_bytes=32 * 2**30, emergency_commit_headroom_bytes=16 * 2**30,
        gpu_index=0, gpu_uuid='GPU-b62da557-0167-4b14-25ef-7b6eab536ef9',
        minimum_gpu_free_bytes=14 * 2**30, emergency_gpu_free_bytes=4 * 2**30,
        controller_yield_seconds=35)
    for name, count in [('grin_original', 10), ('transfer', 270)]:
        if args.only and name != args.only:
            continue
        queue_path = Path('configs/experiments/fm_' + name + '_queue.v1.json')
        queue = json.loads((ROOT / queue_path).read_text(encoding='utf-8'))
        assert len(queue['jobs']) == count
        sources = set(common)
        sources.add(queue_path.as_posix())
        if queue.get('requires_complete_queue'):
            sources.add(queue['requires_complete_queue'])
        budgets = {}
        runners = set()
        for job in queue['jobs']:
            config_path = ROOT / job['config']
            assert digest(config_path) == job['config_sha256']
            config = json.loads(config_path.read_text(encoding='utf-8'))
            # CFMI's original pinned wrapper explicitly requires --trainer.accelerator gpu;
            # its frozen configs have no device field. Do not add a synthetic override.
            cuda = (config.get('device') == 'cuda:0' or
                    job['runner'] == 'scripts/flow_matching/run_cfmi.py' and 'device' not in config)
            assert not config['diagnostic'] and cuda
            sources.update([job['config'], job['runner']])
            runners.add(job['runner'])
            source = ROOT / config['author_source']
            for path in source.rglob('*'):
                if path.is_file() and path.suffix in ('.py', '.yaml', '.yml'):
                    sources.add(path.relative_to(ROOT).as_posix())
            for key in ('data_config', 'dataset_config', 'train_config', 'eval_config'):
                if config.get(key):
                    sources.add(config[key])
            if config.get('paper_id') == 'grin':
                author = ROOT / config['author_config']
                assert digest(author) == config['author_config_sha256']
                sources.add(config['author_config'])
                budgets[config['id']] = yaml.safe_load(author.read_text(encoding='utf-8'))
        # Author wrappers invoke these full evaluation entry points for transfers.
        if name == 'transfer':
            sources.update(['scripts/flow_matching/evaluate_cfmi_transfer.py',
                            'scripts/flow_matching/compare_transfer.py'])
        runtime = dict(schema_version=1, queue=queue_path.as_posix(), queue_sha256=digest(ROOT / queue_path),
            full_job_count=count, allowed_runners=sorted(runners), author_budgets=budgets,
            original_queue_lock='experiments/runs/flow_matching_campaign/queue.lock',
            resource_lock='experiments/runs/fm_grasp_protocol_gpu.lock', settings=settings,
            state_root='experiments/runs/fm_' + name + '_native_resume_v1',
            source_receipts=[dict(path=path, sha256=digest(ROOT / path)) for path in sorted(sources)],
            scope='All existing frozen jobs with unchanged model commands, original phase gates, full datasets, seeds and author budgets.',
            boundary='Scheduler registration is not benchmark performance. Original runner results require a separate independent metric/checkpoint audit before publication. Transfer waits for the original campaign scope review, without waiving its gate.')
        output = ROOT / ('configs/runtime/fm_' + name + '_native_resume.v1.json')
        write_new(output, runtime)
        print(json.dumps(dict(runtime=output.relative_to(ROOT).as_posix(), full_jobs=count, source_bindings=len(sources))))


if __name__ == '__main__':
    main()
