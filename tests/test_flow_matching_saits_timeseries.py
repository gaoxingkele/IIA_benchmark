import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from scripts.flow_matching.audit_author_code import saits_corrected_protocol_copy
from scripts.flow_matching.compare_saits_timeseries import compare
from scripts.flow_matching.prepare_saits_physio import ROOT, sha
from scripts.flow_matching.run_saits import aggregate_groups
from scripts.flow_matching.run_saits_preprocessor import corrected_window_truncate


def test_complete_window_enumeration_preserves_final_full_window_and_drops_partial():
    data = np.arange(72).reshape(72, 1)
    windows = corrected_window_truncate(data, 24)
    assert windows.shape == (3, 24, 1)
    assert np.array_equal(windows[-1, :, 0], data[-24:, 0])
    assert corrected_window_truncate(data[:70], 24).shape == (2, 24, 1)
    assert corrected_window_truncate(data[:20], 24).shape == (0, 24, 1)
    assert corrected_window_truncate(data, 24, 12).shape == (5, 24, 1)


def test_registered_protocol_fixes_actual_author_window_and_mask_functions():
    original = ROOT / 'experiments/runs/flow_matching_campaign/sources/saits/corrected/dataset_generating_scripts/data_processing_utils.py'
    if not original.exists():
        pytest.skip('Optional pinned author source')
    manifest = saits_corrected_protocol_copy()
    modules = []
    for name, path in [('legacy', original), ('fixed', ROOT / manifest['working_path'] / 'dataset_generating_scripts/data_processing_utils.py')]:
        spec = importlib.util.spec_from_file_location('saits_' + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    legacy, fixed = modules
    values = np.arange(72).reshape(72, 1)
    assert len(legacy.window_truncate(values, 24)) == 2
    assert len(fixed.window_truncate(values, 24)) == 3
    vector = np.ones(1000)
    vector[:250] = np.nan
    np.random.seed(26)
    indices = fixed.random_mask(vector, 0.9)
    assert len(indices) == len(np.unique(indices)) == 675
    assert not np.isnan(vector[indices]).any()
    np.random.seed(26)
    legacy_indices = legacy.random_mask(vector, 0.9)
    assert len(np.unique(legacy_indices)) < len(legacy_indices)


def test_month_block_aggregation_preserves_coordinate_weighting():
    groups, totals = aggregate_groups(np.array([[1, 2, 3, 4], [10, 20, 30, 40], [2, 4, 6, 8]]), ['2011-01', '2011-02', '2011-01'])
    assert groups.tolist() == ['2011-01', '2011-02']
    assert totals.tolist() == [[3, 6, 9, 12], [10, 20, 30, 40]]


def test_real_time_series_audits_preserve_disjoint_months_and_exact_missing_rates():
    settings = json.loads((ROOT / 'configs/data/fm_saits_timeseries_original.v1.json').read_text(encoding='utf-8'))
    base = ROOT / settings['processed_root']
    if not (base / 'ett_corrected_windows_exact_masks_missing0.1/data_audit.json').exists():
        pytest.skip('Optional fully prepared original data')
    for dataset in settings['datasets']:
        for ratio in dataset['missing_ratios']:
            root = base / f'{dataset["id"]}_corrected_windows_exact_masks_missing{ratio:g}'
            audit = json.loads((root / 'data_audit.json').read_text(encoding='utf-8'))
            assert audit['total_samples'] == dataset['expected_samples']
            assert audit['mask_protocol'] == 'exact_without_replacement'
            assert abs(audit['splits']['test']['realized_missing_rate'] - ratio) < 1e-5
            months = [set(audit['splits'][split]['months']) for split in ('train', 'val', 'test')]
            assert not months[0] & months[1] and not months[0] & months[2] and not months[1] & months[2]
    legacy = json.loads((base / 'electricity_missing0.9/data_audit.json').read_text(encoding='utf-8'))
    assert 0.59 < legacy['splits']['test']['realized_missing_rate'] < 0.60


def test_time_series_queue_and_hyperparameters_are_frozen_without_ett_best_substitution():
    queue_path = ROOT / 'configs/experiments/fm_saits_timeseries_queue.v1.json'
    queue = json.loads(queue_path.read_text(encoding='utf-8'))
    assert len(queue['jobs']) == 210
    variants = []
    for job in queue['jobs']:
        path = ROOT / job['config']
        assert sha(path) == job['config_sha256']
        config = json.loads(path.read_text(encoding='utf-8'))
        assert sha(ROOT / config['author_config']) == config['author_config_sha256']
        variants.append(config['variant'])
        if config['dataset'] == 'ett':
            assert config['method'] == 'saits_base'
    assert variants.count('author') == variants.count('corrected') == 105
    record = compare(queue_path, ROOT / 'configs/reproducibility/saits/timeseries_references.v1.json')
    assert all(r['verdict'] == 'incomplete' and 'mean' not in r for r in record['records'] if len(r['completed_seeds']) < 5)


def test_formal_comparison_rejects_diagnostic_results(tmp_path):
    configuration = dict(dataset='ett', method='saits_base', variant='author', missing_ratio=0.1, seed=26, output_root='result', id='job')
    path = tmp_path / 'job.json'
    path.write_text(json.dumps(configuration))
    queue_path = tmp_path / 'queue.json'
    queue_path.write_text(json.dumps({'jobs': [{'config': 'job.json', 'config_sha256': sha(path)}]}))
    refs = tmp_path / 'refs.json'
    refs.write_text('{}')
    result = tmp_path / 'result'
    result.mkdir()
    (result / 'result.json').write_text(json.dumps({'status': 'completed', 'config_sha256': sha(path), 'config': {'diagnostic': True}}))
    with pytest.raises(ValueError, match='diagnostic'):
        compare(queue_path, refs, root=tmp_path)


def test_git_preserves_frozen_configuration_bytes_across_line_endings():
    path = 'configs/experiments/fm_original_queue.v1.json'
    result = subprocess.run(['git', '-C', str(ROOT), 'check-attr', 'text', '--', path], capture_output=True, text=True, check=True)
    assert result.stdout.strip().endswith('text: unset')
