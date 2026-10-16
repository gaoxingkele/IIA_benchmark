from scripts.flow_matching import preflight_otcfm_source_cnf_full_inputs_v1 as old
from scripts.flow_matching import preflight_otcfm_source_cnf_full_inputs_v2 as s


def test_native_failure_is_terminal_and_preserved_without_guard_abort():
    p = s.ROOT / 'experiments/runs/fm_otcfm_source_cnf_all12_full_input_preflight_v1'
    receipt = s.read(p / 'guard_receipt.json')
    assert receipt['exit_code'] == 1 and not receipt['passed']
    assert not receipt['resource_guard_aborted'] and receipt['report_sha256'] is None
    assert 'The size of tensor a (1024) must match the size of tensor b (5)' in (p / 'worker.stderr.log').read_text()


def test_every_full_case_solver_batch_optimizer_and_guard_is_retained():
    a, b = old.registration(), s.registration()
    for key in ['cases', 'settings', 'resource_lock', 'environment_lock', 'python',
                'training_batch', 'dimensions', 'seed', 'optimizer', 'expected_parameters',
                'augmented_solver', 'train_val_test_split', 'original_scope']:
        assert a[key] == b[key]
    assert len(b['cases']) == 12 and b['training_batch'] == 1024
    assert b['optimizer_updates'] == 0 and not b['all_original_experiments_complete']
    assert b['state_root'] != a['state_root'] and b['artifact_root'] != a['artifact_root']


def test_trace_repair_and_original_augmentation_file_are_bound():
    c = s.registration()
    paths = {v['path'] for v in c['source_receipts']}
    assert 'src/iia_benchmark/models/otcfm_cnf_trace_shape_v1.py' in paths
    assert 'tests/test_fm_otcfm_cnf_trace_shape_v1.py' in paths
    assert f'{s.source.AUTHOR}/runner/src/models/components/augmentation.py' in paths
    profile = s.read(s.ROOT / s.MODEL)
    assert profile['likelihood_trace'] == 'exact' and profile['expected_parameters'] == 9093
    assert not profile['historical_paper_equivalence_certified'] and not profile['full_training_complete']
