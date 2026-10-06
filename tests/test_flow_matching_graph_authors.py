import hashlib
import json
from pathlib import Path

import pytest

from scripts.flow_matching.compare_grin import compare

ROOT = Path(__file__).resolve().parents[1]


def test_grin_jobs_freeze_five_seeds_author_yaml_and_both_graph_datasets():
    queue = json.loads((ROOT / 'configs/experiments/fm_grin_original_queue.v1.json').read_text(encoding='utf-8'))
    assert len(queue['jobs']) == 10
    assert queue['python'] == '.venv/grin-author-py38/Scripts/python.exe'
    datasets = {}
    for job in queue['jobs']:
        path = ROOT / job['config']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == job['config_sha256']
        cfg = json.loads(path.read_text(encoding='utf-8'))
        assert not cfg['diagnostic'] and cfg['device'] == 'cuda:0'
        datasets.setdefault(cfg['dataset'], set()).add(cfg['seed'])
        author_yaml = ROOT / cfg['author_config']
        if author_yaml.exists():
            assert hashlib.sha256(author_yaml.read_bytes()).hexdigest() == cfg['author_config_sha256']
    assert datasets == {'air36': set(range(2026, 2031)), 'air': set(range(2026, 2031))}


def test_graph_cpu_preflights_cover_both_datasets_and_exact_epoch_resume():
    base = ROOT / 'experiments/runs/flow_matching_campaign'
    if not (base / 'grin_cpu_preflight_report.json').exists():
        pytest.skip('Optional local author environments/data')
    grin = json.loads((base / 'grin_cpu_preflight_report.json').read_text(encoding='utf-8'))
    assert grin['status'] == 'passed' and grin['diagnostic'] is True
    assert all(grin['checks'].values())
    assert set(grin['datasets']) == {'air36', 'air'}
    for record in grin['datasets'].values():
        assert not any(record['data_audit']['timestamp_overlap_counts'].values())
        assert record['data_audit']['eval_targets_unknown'] == 0
    giflow = json.loads((base / 'giflow_cpu_preflight/report.json').read_text(encoding='utf-8'))
    assert giflow['status'] == 'passed' and giflow['diagnostic'] is True
    assert giflow['nodes'] == 36 and giflow['sample_finite'] and giflow['training_loss_finite']


def comparison_fixture(root, completed, diagnostic=False):
    reference = {'runs': 5, 'boundary': 'numeric only', 'datasets': {'air36': {
        'mae': {'mean': 12.08, 'reported_plus_minus': .47, 'rounding_half_unit': .005}}}}
    jobs = []
    for seed in range(5):
        cfg = {'id': str(seed), 'dataset': 'air36', 'seed': seed, 'output_root': 'runs/' + str(seed)}
        path = root / (str(seed) + '.json')
        path.write_text(json.dumps(cfg), encoding='utf-8')
        job = {'config': path.name, 'config_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        jobs.append(job)
        if seed < completed:
            directory = root / cfg['output_root']
            directory.mkdir(parents=True)
            (directory / 'result.json').write_text(json.dumps({'status': 'completed', 'config_sha256': job['config_sha256'],
                'config': cfg, 'diagnostic': diagnostic, 'metrics': {'mae': 12.08 + seed / 10}}), encoding='utf-8')
    queue = root / 'queue.json'
    refs = root / 'reference.json'
    queue.write_text(json.dumps({'jobs': jobs}), encoding='utf-8')
    refs.write_text(json.dumps(reference), encoding='utf-8')
    return queue, refs


def test_grin_partial_seeds_do_not_produce_complete_comparison(tmp_path):
    queue, refs = comparison_fixture(tmp_path, 4)
    report = compare(queue, refs, tmp_path)
    assert report['records'][0]['verdict'] == 'incomplete'
    assert 'mean' not in report['records'][0]


def test_grin_comparison_includes_every_seed_and_does_not_claim_equivalence(tmp_path):
    queue, refs = comparison_fixture(tmp_path, 5)
    report = compare(queue, refs, tmp_path)
    row = report['records'][0]
    assert row['mean'] == pytest.approx(12.28)
    assert row['verdict'] == 'numeric_comparison_only_protocol_alignment_unverified'
    assert row['standard_error'] > 0


def test_grin_diagnostic_results_are_rejected(tmp_path):
    queue, refs = comparison_fixture(tmp_path, 5, diagnostic=True)
    with pytest.raises(ValueError, match='diagnostic'):
        compare(queue, refs, tmp_path)


def test_graph_reference_mre_percent_conversion():
    reference = json.loads((ROOT / 'configs/reproducibility/grin_references.v1.json').read_text(encoding='utf-8'))
    assert reference['datasets']['air36']['mre']['mean'] == .17
    assert reference['datasets']['air']['mre']['mean'] == .2182
    assert reference['pdf_page'] == 7 and reference['setting'] == 'out_of_sample'
