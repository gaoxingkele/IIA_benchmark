"""Audit full GRASP benchmark inputs and freeze native statistical replays."""
import argparse
import json
from pathlib import Path
import subprocess

from scripts.flow_matching.mtsbench_stat_protocol import ROOT, sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/fm_grasp_stat_registration.v1.json')
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding='utf-8'))
    output = ROOT / config['queue_output']
    if output.exists():
        raise FileExistsError('Registered queue is immutable')
    import pandas as pd
    manifest = json.loads((ROOT / config['data_manifest']).read_text(encoding='utf-8'))
    datasets = {d['id']: d for d in manifest['datasets']}
    audited, jobs = [], []
    for spec in config['datasets']:
        dataset = datasets[spec['id']]
        if len(dataset['entities']) != spec['entities']:
            raise ValueError('Paper entity count mismatch')
        sources = {s['path']: s for s in dataset['source_files']}
        rows = []
        for entity in dataset['entities']:
            entity_id = entity['entity_id']
            paths = {kind: next(s for p, s in sources.items() if p.endswith('/' + entity_id + '_' + kind + '.csv'))
                     for kind in ('train', 'test')}
            for kind, source in paths.items():
                if sha(ROOT / source['path']) != source['sha256']:
                    raise ValueError('Registered raw data changed')
            headers = pd.read_csv(ROOT / paths['test']['path'], nrows=0).columns.tolist()
            train_headers = pd.read_csv(ROOT / paths['train']['path'], nrows=0).columns.tolist()
            train_features = [c for c in train_headers if c not in ('timestamp', 'is_anomaly')]
            if (train_features != headers[1:-1] or train_headers[0] != 'timestamp'
                    or headers[0] != 'timestamp' or headers[-1] != 'is_anomaly'):
                raise ValueError('Native feature-column contract differs')
            rows.append({'id': entity_id, 'test_path': paths['test']['path'], 'test_sha256': paths['test']['sha256'],
                         'train_path': paths['train']['path'], 'train_sha256': paths['train']['sha256'],
                         'feature_names': headers[1:-1], 'features': len(headers)-2,
                         'paper_variables': spec['paper_variables'], 'training_anomaly_points': entity['training_anomaly_points'],
                         'native_fit_role': 'test_features_only', 'labels_used_for_fitting': False})
        audited.append({'dataset': spec['id'], 'paper_entities': spec['entities'], 'entities': rows,
                        'paper_variables': spec['paper_variables'], 'observed_feature_counts': sorted({e['features'] for e in rows}),
                        'feature_count_discrepancy': any(e['features'] != spec['paper_variables'] for e in rows)})
        for seed in config['seeds']:
            for model in config['algorithms']:
                ident = f"grasp_mtsbench_native__{model['id'].lower()}__{spec['id']}__seed{seed}"
                jobs.append({'id': ident, 'algorithm': model['id'], 'dataset': spec['id'], 'seed': seed,
                             'parameters': model['parameters'], 'model_config': model['model_config'],
                             'entities': rows, 'protocol': config['protocol'],
                             'output_directory': config['state_root'] + '/jobs/' + ident})
    lock_path = ROOT / config['environment_lock']
    if lock_path.exists():
        raise FileExistsError('Frozen environment lock exists')
    command = [str(ROOT / config['python']), '-c',
               "import json,importlib.metadata; print(json.dumps({d.metadata['Name'].lower():importlib.metadata.version(d.metadata['Name']) for d in importlib.metadata.distributions()}))"]
    environment = json.loads(subprocess.check_output(command, cwd=ROOT, text=True))
    write_json(lock_path, environment)
    source_paths = [str(p.relative_to(ROOT)).replace('\\', '/') for p in (ROOT / config['author_source']).rglob('*')
                    if p.is_file() and '__pycache__' not in p.parts]
    source_paths += [args.config, config['acquisition_config'], config['data_manifest'], config['environment_lock'], config['paper_pdf'],
                     'scripts/flow_matching/mtsbench_stat_protocol.py', 'scripts/flow_matching/run_mtsbench_stat_queue.py',
                     'scripts/flow_matching/register_grasp_stat.py']
    source_paths += [m['model_config'] for m in config['algorithms']]
    queue = {k: config[k] for k in ('author_source', 'state_root', 'artifact_root', 'python', 'environment_lock', 'runtime_config', 'settings', 'boundary')}
    queue.update(schema_version=1, jobs=jobs, data_audit=audited,
                 source_receipts=[{'path': p, 'sha256': sha(ROOT / p)} for p in sorted(set(source_paths))])
    write_json(output, queue)
    print(json.dumps({'jobs': len(jobs), 'datasets': [(d['dataset'], len(d['entities']), d['observed_feature_counts']) for d in audited],
                      'source_receipts': len(queue['source_receipts']), 'environment_packages': len(environment)}))


if __name__ == '__main__':
    main()
