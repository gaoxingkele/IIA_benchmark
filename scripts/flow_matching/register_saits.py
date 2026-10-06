"""Freeze SAITS-paper original PhysioNet experiments and diagnostic configurations."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.register_transfer import write_frozen


def main():
    dataset = json.loads((ROOT / 'configs/data/fm_saits_physio_original.v1.json').read_text(encoding='utf-8'))
    jobs = []
    for seed in (26, 27, 28, 29, 30):
        for model in ('SAITS', 'BRITS'):
            original = ROOT / f'experiments/runs/flow_matching_campaign/sources/saits/original/configs/PhysioNet2012_{model}_best.ini'
            ini = ROOT / f'configs/reproducibility/saits/PhysioNet2012_{model}_best.ini'
            ini.parent.mkdir(parents=True, exist_ok=True)
            if ini.exists() and ini.read_bytes() != original.read_bytes():
                raise ValueError('Existing frozen hyperparameters changed; preserve them')
            if not ini.exists():
                ini.write_bytes(original.read_bytes())
            identifier = f'saits_paper_{model.lower()}_physio_seed{seed}'
            config = {'schema_version': 1, 'id': identifier, 'method': model.lower(), 'phase': 'author_original_protocol',
                      'paper_id': 'saits', 'seed': seed, 'device': 'cuda', 'diagnostic': False,
                      'author_config': ini.relative_to(ROOT).as_posix(),
                      'author_config_sha256': hashlib.sha256(ini.read_bytes()).hexdigest(),
                      'dataset_root': dataset['output_root'], 'output_root': f'experiments/runs/flow_matching_campaign/{identifier}',
                      'citation': dataset['literature_citation'], 'data_citation': dataset['data_citation'],
                      'boundary': 'BRITS here is the SAITS author baseline, not reproduction of the original BRITS 2018 protocol. Historical split order and worker random streams cannot be exactly aligned.'}
            path = ROOT / f'configs/experiments/fm_saits_jobs/{identifier}.json'
            write_frozen(path, config)
            jobs.append({'config': path.relative_to(ROOT).as_posix(), 'runner': 'scripts/flow_matching/run_saits.py',
                         'config_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            if seed == 26:
                for kind in ('full', 'resumed'):
                    diagnostic = dict(config, id=f'saits_{model.lower()}_cpu_{kind}', device='cpu', diagnostic=True,
                                      diagnostic_epochs=2, output_root=f'experiments/runs/flow_matching_campaign/saits_{model.lower()}_cpu_{kind}')
                    write_frozen(ROOT / f'configs/experiments/fm_saits_jobs/diagnostic_{model.lower()}_{kind}.json', diagnostic)
    queue = {'schema_version': 1, 'id': 'fm_saits_physio_original_queue_v1', 'python': '.venv/fm-author-py310/Scripts/python.exe',
             'requires_complete_queue': 'configs/experiments/fm_air36_original_queue.v1.json', 'jobs': jobs}
    write_frozen(ROOT / 'configs/experiments/fm_saits_physio_original_queue.v1.json', queue)
    print(json.dumps({'jobs': len(jobs), 'diagnostics': 4, 'prerequisite': queue['requires_complete_queue']}))


if __name__ == '__main__':
    main()
