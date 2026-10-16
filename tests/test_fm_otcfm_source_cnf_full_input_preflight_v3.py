from scripts.flow_matching import preflight_otcfm_source_cnf_full_inputs_v2 as old
from scripts.flow_matching import preflight_otcfm_source_cnf_full_inputs_v3 as s


def test_actual_memory_abort_is_preserved_and_not_misread_as_timeout():
    path=s.ROOT/'experiments/runs/fm_otcfm_source_cnf_all12_full_input_preflight_v2'
    guard=s.read(path/'guard_receipt.json');progress=s.read(path/'progress.json')
    assert not guard['passed'] and guard['resource_guard_aborted'] and guard['exit_code']!=0
    assert progress['completed_cases']==4 and progress['required_cases']==12
    assert guard['peak_tree_private_bytes']>8*2**30


def test_all_original_data_batch_precision_optimizer_and_guards_are_unchanged():
    a,b=old.registration(),s.registration()
    for key in ['cases','training_batch','dimensions','settings','augmented_solver','optimizer',
                'resource_lock','python','environment_lock','train_val_test_split','seed','original_scope']:
        assert a[key]==b[key]
    assert len(b['cases'])==12 and b['training_batch']==1024 and b['optimizer_updates']==0
    assert b['settings']['maximum_worker_private_bytes']==8*2**30
    assert b['state_root']!=a['state_root'] and b['artifact_root']!=a['artifact_root']


def test_complete_endpoint_backward_is_bound_and_original_exact_trace_remains():
    c=s.registration();paths={r['path'] for r in c['source_receipts']}
    assert 'src/iia_benchmark/models/otcfm_cnf_endpoint_backward_v1.py' in paths
    assert 'tests/test_fm_otcfm_cnf_endpoint_backward_v1.py' in paths
    profile=s.read(s.ROOT/s.MODEL)
    assert profile['likelihood_trace']=='exact' and profile['expected_parameters']==9093
    assert not profile['full_training_complete'] and not profile['historical_paper_equivalence_certified']
