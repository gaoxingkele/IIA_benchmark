import ast

from scripts.flow_matching import replay_shcl_native_frames_v1 as replay


def test_exact_author_adjustment_statements_extracted_without_fixing_index_zero_quirk():
    model=replay.native.read(replay.native.ROOT/'configs/models/fm_shcl_released_vae_native.v1.json')
    source=(replay.native.verify(model)/'solver.py').read_text(encoding='utf-8')
    tree=ast.parse(source); solver=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Solver')
    function=next(n for n in solver.body if isinstance(n,ast.FunctionDef) and n.name=='test')
    start=next(i for i,n in enumerate(function.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='anomaly_state' for t in n.targets))
    end=next(i for i,n in enumerate(function.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='pa_accuracy' for t in n.targets))
    extracted=replay.author_adjustment_ast(source).body[0].body[:-1]
    assert [ast.dump(n) for n in extracted]==[ast.dump(n) for n in function.body[start:end]]


def test_original_first_event_backward_boundary_preserved_separately_from_strict_TAB():
    import numpy as np
    model=replay.native.read(replay.native.ROOT/'configs/models/fm_shcl_released_vae_native.v1.json')
    namespace=dict(np=np)
    source=(replay.native.verify(model)/'solver.py').read_text(encoding='utf-8')
    exec(compile(replay.author_adjustment_ast(source),'<original_adjustment_test>','exec'),namespace)
    prediction,truth=namespace['adjustment'](np.array([0,0,1,0]),np.array([1,1,1,0]))
    assert prediction.tolist()==[0,1,1,0]
    assert truth.tolist()==[1,1,1,0]


def test_bitwise_replay_registration_keeps_existing_memory_caps_and_full_fixture_frames():
    cfg=replay.native.read(replay.native.ROOT/'configs/runtime/fm_shcl_independent_replay_fixture.v1.json')
    old=replay.native.read(replay.native.ROOT/'configs/runtime/fm_shcl_native_observer_audit.v1.json')
    assert cfg['settings']==old['settings'] and cfg['resource_lock']==old['resource_lock']
    assert cfg['full_native_frames_required']==2 and cfg['full_native_windows_required']==128
    assert cfg['bitwise_array_and_post_RNG_gate']
    assert cfg['replayed_array_root'].startswith('F:/aicoding/IIA_Data/')


def test_actual_complete_saved_fixture_weight_optimizer_and_every_stochastic_frame_replayed():
    cfg=replay.native.read(replay.native.ROOT/'configs/runtime/fm_shcl_independent_replay_fixture.v1.json')
    report=replay.native.read(replay.native.ROOT/cfg['output_root']/'full_observer_fidelity_fixture.json')
    assert report['passed'] and len(report['training_checkpoints'])==1
    assert report['training_checkpoints'][0]['all_model_and_Adam_state_bitwise_restored']
    assert len(report['full_native_frames_replayed'])==2
    assert sum(row['frame']['shape'][0] for row in report['full_native_frames_replayed'])==128
    for row in report['full_native_frames_replayed']:
        assert row['all_full_native_losses_scores_labels_bitwise_verified']
        assert row['complete_original_CPU_NumPy_Python_CUDA_RNG_poststate_verified']
        assert row['optimizer_updates_executed']==0
    assert report['all_source_original_metrics_exactly_verified'] and report['all_model_variables_unchanged']
    assert report['synthetic_operator_only_not_benchmark_performance'] and report['original_benchmark_optimizer_updates']==0


def test_full_completion_consumer_contains_every_actual_canonical_command_and_never_fixture_slots():
    cfg=replay.native.read(replay.native.ROOT/'configs/runtime/fm_shcl_full413_completion_replay.v1.json')
    queue=replay.native.read(replay.native.ROOT/cfg['original_queue'])
    assert cfg['canonical_job_ids']==[j['id'] for j in queue['jobs']]
    assert len(cfg['canonical_job_ids'])==413
    assert cfg['full_training_commands']==49 and cfg['full_test_commands']==364
    assert not any('fixture' in value for value in cfg['canonical_job_ids'])
    assert cfg['bitwise_loss_score_label_and_RNG_gate'] and cfg['source_test_prefix_not_strict_full_point_TAB']
    assert cfg['every_original_epoch_model_and_Adam_restore_required']
