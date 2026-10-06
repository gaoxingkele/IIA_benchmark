"""Freeze original and corrected SAITS-paper time-series experiment queues."""
from configparser import ConfigParser, ExtendedInterpolation
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.audit_author_code import saits_corrected_protocol_copy
from scripts.flow_matching.prepare_saits_physio import sha
from scripts.flow_matching.register_transfer import write_frozen


def freeze_bytes(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise ValueError('Frozen author config changed; preserved')
    if not path.exists():
        path.write_bytes(payload)


def model_ini(dataset, method, audit):
    original = ROOT / 'experiments/runs/flow_matching_campaign/sources/saits/original/configs'
    target = ROOT / f'configs/reproducibility/saits/{dataset}_{method}_best.ini'
    if dataset == 'ett':
        # Best ETT configuration is absent. Reconstruct only the explicitly
        # specified SAITS-base architecture from Appendix A and author base INI.
        source = original / 'PhysioNet2012_SAITS_base.ini'
        parser = ConfigParser(interpolation=ExtendedInterpolation())
        parser.read(source, encoding='utf-8')
        parser['dataset'].update(dataset_name='ett', seq_len='24', feature_num='7',
                                 eval_every_n_steps=str(math.ceil(audit['splits']['train']['windows'] / 128)))
        parser['model']['model_name'] = 'ETT_SAITS_base_preregistered'
        buffer = io.StringIO()
        parser.write(buffer)
        target = ROOT / 'configs/reproducibility/saits/ETT_SAITS_base_reconstructed.ini'
        freeze_bytes(target, buffer.getvalue().encode('utf-8'))
    else:
        prefix = 'AirQuality' if dataset == 'air_quality_132' else 'Electricity'
        source = original / f'{prefix}_{method.upper()}_best.ini'
        freeze_bytes(target, source.read_bytes())
    return target


def main():
    config_path = ROOT / 'configs/data/fm_saits_timeseries_original.v1.json'
    settings = json.loads(config_path.read_text(encoding='utf-8'))
    saits_corrected_protocol_copy()
    jobs = []
    for corrected in (False, True):
        variant = 'corrected' if corrected else 'author'
        for dataset in settings['datasets']:
            methods = ('saits_base',) if dataset['id'] == 'ett' else ('saits', 'brits')
            for ratio in dataset['missing_ratios']:
                suffix = '_corrected_windows_exact_masks' if corrected else ''
                dataset_root = f'{settings["processed_root"]}/{dataset["id"]}{suffix}_missing{ratio:g}'
                audit = json.loads((ROOT / dataset_root / 'data_audit.json').read_text(encoding='utf-8'))
                for method in methods:
                    ini = model_ini(dataset['id'], method, audit)
                    for seed in (26, 27, 28, 29, 30):
                        identifier = f'saits_paper_{method}_{dataset["id"]}_{variant}_missing{ratio:g}_seed{seed}'
                        config = {'schema_version': 1, 'id': identifier, 'paper_id': 'saits', 'method': method,
                                  'dataset': dataset['id'], 'missing_ratio': ratio, 'seed': seed, 'device': 'cuda',
                                  'phase': 'corrected_independent_test' if corrected else 'author_original_protocol',
                                  'variant': variant, 'corrected_masking': corrected, 'bootstrap_group': 'month', 'diagnostic': False,
                                  'author_config': ini.relative_to(ROOT).as_posix(), 'author_config_sha256': sha(ini),
                                  'dataset_root': dataset_root, 'dataset_sha256': audit['dataset_sha256'],
                                  'output_root': f'experiments/runs/flow_matching_campaign/{identifier}',
                                  'citation': settings['literature_citation'], 'data_citation': dataset['data_citation'],
                                  'boundary': 'ETT SAITS-base architecture specified in paper Appendix A; dataset settings and per-epoch validation reconstructed because author ETT INI is missing.' if dataset['id'] == 'ett' else
                                      'Frozen released best configuration reused across missing rates; exact historical per-rate tuning and preprocessing seeds unavailable. Corrected and released protocols compared separately.'}
                        path = ROOT / f'configs/experiments/fm_saits_timeseries_jobs/{identifier}.json'
                        write_frozen(path, config)
                        jobs.append({'config': path.relative_to(ROOT).as_posix(), 'config_sha256': sha(path),
                                     'runner': 'scripts/flow_matching/run_saits.py'})
                        if seed == 26 and ratio == 0.1:
                            diagnostic = dict(config, id=f'saits_ts_{dataset["id"]}_{method}_{variant}_cpu', device='cpu',
                                diagnostic=True, diagnostic_epochs=2,
                                output_root=f'experiments/runs/flow_matching_campaign/saits_ts_{dataset["id"]}_{method}_{variant}_cpu')
                            write_frozen(ROOT / f'configs/experiments/fm_saits_timeseries_jobs/diagnostic_{dataset["id"]}_{method}_{variant}.json', diagnostic)
                            if dataset['id'] == 'electricity' and method == 'saits' and corrected:
                                resumed = dict(diagnostic, id=diagnostic['id'] + '_resumed', output_root=diagnostic['output_root'] + '_resumed')
                                write_frozen(ROOT / 'configs/experiments/fm_saits_timeseries_jobs/diagnostic_electricity_saits_corrected_resumed.json', resumed)
    queue = {'schema_version': 1, 'id': 'fm_saits_timeseries_queue_v1', 'python': '.venv/fm-author-py310/Scripts/python.exe',
             'requires_complete_queue': 'configs/experiments/fm_saits_physio_original_queue.v1.json', 'jobs': jobs}
    write_frozen(ROOT / 'configs/experiments/fm_saits_timeseries_queue.v1.json', queue)
    evidence = {'paper_id': 'saits', 'author_commit': '660b87f19c1277065f314f24134f646229e89ca9',
                'author_config_inventory': sorted(p.name for p in (ROOT / 'experiments/runs/flow_matching_campaign/sources/saits/original/configs').glob('*.ini')),
                'ett_best_status': 'exact_best_hyperparameters_not_in_pinned_author_repository',
                'ett_base_status': 'paper_appendix_A_parameters_preregistered_with_reconstructed_dataset_validation_frequency',
                'original_scope_status': 'in_progress_other_ablation_downstream_NRTSI_and_ETT_best_not_completed'}
    (ROOT / 'docs/reports/flow_matching_saits_protocol_availability.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps({'jobs': len(jobs), 'variants': ['author', 'corrected'], 'datasets': [d['id'] for d in settings['datasets']]}))


if __name__ == '__main__':
    main()
