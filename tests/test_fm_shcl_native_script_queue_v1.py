import math

from scripts.flow_matching.run_shcl_native_job_v1 import observer,verify


def config():
    return observer.native.read(observer.native.ROOT/'configs/experiments/fm_shcl_native_script_queue.v1.json')


def test_every_original413_command_native_workers_seed_and_full_budget_preserved():
    queue=config(); verify(queue)
    assert len(queue['jobs'])==413
    train=[j for j in queue['jobs'] if j['mode']=='train']; tests=[j for j in queue['jobs'] if j['mode']=='test']
    assert len(train)==49 and len(tests)==364
    assert sum(j['num_epochs'] for j in train)==155
    assert {j['seed'] for j in queue['jobs']}=={1}
    assert {j['native_DataLoader_workers'] for j in queue['jobs']}=={8}
    assert not any(j['synthetic_fixture'] for j in queue['jobs'])
    for job in tests:
        parent=next(j for j in train if j['id']==job['depends_on'])
        assert job['checkpoint_directory']==parent['checkpoint_directory']
        assert (job['dataset'],job['win_size'],job['num_epochs'])==(parent['dataset'],parent['win_size'],parent['num_epochs'])
        assert job['output_directory']!=parent['output_directory']


def test_actual_full_size_observer_fixture_preserves_all_parameters_Adam_state_and_rng_without_benchmark_credit():
    cfg=observer.native.read(observer.native.ROOT/'configs/runtime/fm_shcl_native_observer_audit.v1.json')
    proof=observer.native.read(observer.native.ROOT/cfg['output_root']/'full_observer_fidelity_fixture.json')
    assert proof['passed'] and proof['full_input_shape']==[64,110,25]
    assert proof['all_native_parameters_and_Adam_state_bitwise_unchanged_by_observation']
    assert proof['source_loss_and_parent_Torch_RNG_bitwise_equal']
    assert proof['serialized_checkpoint_and_all_frame_observation_RNG_unchanged']
    assert proof['synthetic_optimizer_updates']==1 and proof['original_benchmark_optimizer_updates']==0
    assert proof['synthetic_operator_only_not_benchmark_performance'] and proof['new_benchmark_results']==0


def test_actual_native_Torch111_operator_and_complete52_data_modes_verified():
    for kind in ('operators','data'):
        cfg=observer.native.read(observer.native.ROOT/('configs/runtime/fm_shcl_author_'+kind+'.v1.json'))
        path=observer.native.ROOT/cfg['output_root']/'full_released_operator_audit.json' if kind=='operators' else \
            observer.native.ROOT/cfg['view_root']/'full_data_preparation_audit.json'
        report=observer.native.read(path)
        assert report['passed'] and report['environment']['packages']['torch']=='1.11.0+cu113'
        assert report['environment']['packages']['numpy']=='1.26.4'
        assert report['full_operator_profiles']==13 if kind=='operators' else report['full_native_loader_modes_audited']==52
        assert not report['original_training_executed'] and not report.get('paper_performance_reproduced',False)
