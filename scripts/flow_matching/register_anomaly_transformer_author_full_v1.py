"""Freeze the missing full-width Anomaly Transformer author-runtime matrix."""
import json
from pathlib import Path
import sys

from scripts.flow_matching.run_anomaly_transformer_author_full_v1 import ROOT, read, sha, verify


CONFIG = 'configs/experiments/fm_anomaly_transformer_author_full.v1.json'
SOURCE = 'experiments/runs/fm_baseline_code_completion/sources/anomaly_transformer_author/original'


def main():
    output = ROOT / CONFIG
    if output.exists():
        raise FileExistsError('Keep frozen registrations; do not overwrite')
    receipts = []
    tracked = [ROOT / SOURCE / p for p in ('main.py', 'solver.py', 'README.md', 'scripts/Start.sh',
               'scripts/SMD.sh', 'scripts/MSL.sh', 'scripts/SMAP.sh', 'scripts/PSM.sh')]
    tracked += sorted((ROOT / SOURCE / 'model').glob('*.py'))
    tracked += sorted((ROOT / SOURCE / 'data_factory').glob('*.py'))
    tracked += sorted((ROOT / SOURCE / 'utils').glob('*.py'))
    tracked += [Path(__file__), ROOT / 'scripts/flow_matching/run_anomaly_transformer_author_full_v1.py',
                ROOT / 'scripts/flow_matching/heavy_resources.py',
                ROOT / 'scripts/flow_matching/heavy_scheduler_v2.py',
                ROOT / 'tests/test_fm_anomaly_transformer_author_full_v1.py',
                ROOT / 'configs/models/fm_anomaly_transformer_author_full.v1.json',
                ROOT / 'papers/literature/mtsad_pdfs/xu2022_anomaly_transformer.pdf']
    for dataset in ('SMD', 'MSL', 'SMAP', 'PSM'):
        path = ROOT / ('configs/datasets/mtsad_' + dataset.lower() + '.json')
        tracked.append(path)
        dataset_config = read(path)
        tracked += [ROOT / 'data/public_datasets/mtsad' / dataset_config['directory'] / name
                    for name in dataset_config['files'].values()]
    for path in sorted(set(tracked)):
        receipts.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path)))
    jobs = []
    for dataset in ('SMD', 'MSL', 'PSM', 'SMAP', 'SWaT'):
        for seed in (1103, 1104, 1105, 1106, 1107):
            jobs.append(dict(id=f'anomaly_transformer_author__full_width__{dataset.lower()}__seed{seed}',
                dataset=dataset, seed=seed, epochs=10 if dataset=='SMD' else 3,
                anomaly_ratio=0.5 if dataset=='SMD' else (0.1 if dataset=='SWaT' else 1.0),
                dataset_config=f'configs/datasets/mtsad_{dataset.lower()}.json',
                status='source_loader_unavailable' if dataset=='SWaT' else 'registered_full_author_runtime'))
    config = dict(schema_version=1, source_repository='https://github.com/thuml/Anomaly-Transformer',
        source_commit='b0ee470c8012bff857fb600462aec6209c4a18d9', source_directory=SOURCE,
        paper='papers/literature/mtsad_pdfs/xu2022_anomaly_transformer.pdf',
        citation='https://doi.org/10.48550/arXiv.2110.02642',
        python=str(Path(sys.executable)), cpu_threads=2, seeds=[1103,1104,1105,1106,1107], jobs=jobs,
        data_root='data/public_datasets/mtsad',
        artifact_root='F:/aicoding/IIA_Data/experiments/anomaly_transformer_author_full_v1',
        state_root='experiments/runs/fm_anomaly_transformer_author_full_v1',
        resource_lock='experiments/runs/fm_heavy_recovery_cpu.lock',
        settings=dict(minimum_available_memory_bytes=32*1024**3,
                      minimum_commit_headroom_bytes=40*1024**3,
                      emergency_commit_headroom_bytes=16*1024**3),
        required_model=dict(d_model=512, d_ff=512, n_heads=8, e_layers=3, dropout=0.0,
                            window=100, batch_size=256, k=3, learning_rate=0.0001, temperature=50),
        original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
        source_receipts=receipts,
        outstanding_author_fidelity_gaps=['SWaT loader absent from pinned original snapshot',
             'Paper-wide ablation and hyperparameter tables require their own full-width registrations',
             'Independent checkpoint replay and multi-seed uncertainty required after training',
             'Original snapshot is a maintainer version, not certified paper-era commit',
             'Grouped strict TAB protocol is a separate required track'],
        boundary='Full-width pinned author-runtime CPU port; only explicit CUDA/NumPy compatibility and '
             'observation hooks. Original test-based checkpoint selection, train/test pooled calibration, '
             'PA, source data/window strides and omitted test tails retained. Seeds are local registered '
             'repetitions. Existing d_model32/d_ff32 local reconstructions remain distinct; no equivalence '
             'claim, strict TAB certificate, or completion of all45/681/273 records.')
    verify(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(config, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(config=CONFIG, registered_jobs=25, runnable_source_loader_jobs=20,
                          missing_original_SWaT_loader_jobs=5, full_width=512)))


if __name__ == '__main__':
    main()
