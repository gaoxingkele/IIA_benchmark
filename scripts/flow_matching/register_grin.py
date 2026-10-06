"""Register immutable GRIN original Air data jobs and a real-data resume diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_frozen(path, payload):
    text = json.dumps(payload, indent=2)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != payload:
            raise ValueError('Registered configuration changed: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments/fm_grin_original_suite.v1.json')
    args = parser.parse_args()
    suite = json.loads(args.config.read_text(encoding='utf-8'))
    source = ROOT / suite['author_source']
    snapshot = json.loads((source.parent / 'snapshot.json').read_text(encoding='utf-8'))
    data = json.loads((ROOT / suite['data_config']).read_text(encoding='utf-8'))
    audit = json.loads((ROOT / data['output_root'] / 'data_audit.json').read_text(encoding='utf-8'))
    jobs = []
    for dataset, author_yaml in suite['datasets'].items():
        for seed in suite['seeds']:
            identifier = 'grin_original_' + dataset + '_seed' + str(seed)
            config = {'schema_version': 1, 'id': identifier, 'paper_id': 'grin', 'method': 'grin',
                      'dataset': dataset, 'seed': seed, 'device': 'cuda:0', 'diagnostic': False,
                      'author_source': suite['author_source'], 'author_commit': snapshot['commit'],
                      'author_config': (source / author_yaml).relative_to(ROOT).as_posix(),
                      'author_config_sha256': sha(source / author_yaml), 'data_config': suite['data_config'],
                      'data_sha256': audit['files']['small36.h5' if dataset == 'air36' else 'full437.h5']['sha256'],
                      'output_root': suite['output_root'] + '/' + identifier,
                      'metric_protocol': 'grin_original_whole_series_mean_v1', 'citation': suite['citation'],
                      'boundary': suite['boundary']}
            path = ROOT / suite['job_directory'] / (identifier + '.json')
            write_frozen(path, config)
            jobs.append({'config': path.relative_to(ROOT).as_posix(), 'runner': 'scripts/flow_matching/run_grin.py',
                         'config_sha256': sha(path)})
    queue = {'schema_version': 1, 'id': 'fm_grin_original_queue_v1', 'python': suite['python'],
             'requires_complete_queue': suite['requires_complete_queue'], 'jobs': jobs}
    write_frozen(ROOT / suite['queue_path'], queue)
    first = json.loads((ROOT / jobs[0]['config']).read_text(encoding='utf-8'))
    for variant in ('full', 'resumed', 'full_v2', 'resumed_v2'):
        diagnostic = dict(first, id='grin_air36_resume_diagnostic_' + variant,
                          device='cpu', diagnostic=True, diagnostic_epochs=2,
                          output_root=suite['output_root'] + '/grin_air36_resume_diagnostic_' + variant)
        write_frozen(ROOT / suite['job_directory'] / ('diagnostic_' + variant + '.json'), diagnostic)
    full_air = json.loads((ROOT / jobs[5]['config']).read_text(encoding='utf-8'))
    write_frozen(ROOT / suite['job_directory'] / 'diagnostic_full437.json',
                 dict(full_air, id='grin_full437_diagnostic', device='cpu', diagnostic=True, diagnostic_epochs=2,
                      output_root=suite['output_root'] + '/grin_full437_diagnostic'))
    print(json.dumps({'jobs': len(jobs), 'queue': suite['queue_path']}))


if __name__ == '__main__':
    main()
