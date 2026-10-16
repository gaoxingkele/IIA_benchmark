import ast
from scripts.flow_matching import replay_TrajectoryNet_author_fig8_full_v1 as s


def test_every_original_population_checkpoint_and_solver_is_bound():
    c=s.registration();profile=s.read(s.ROOT/s.MODEL)
    assert c['full_input_cells']==16819 and c['original_PC_dimensions']==100 and c['native_dimensions']==5
    assert c['full_population_counts']=={'0':2381,'1':4163,'2':3278,'3':3665,'4':3332}
    assert c['leftout_timepoint']==3 and c['expected_parameters']==9621
    assert c['author_training_seed'] is None and c['local_training_runs_added']==0
    assert profile['source_parameters']['top_k_reg']==.1 and profile['source_parameters']['vecint']==0
    assert profile['full_trajectory_grid_points']==100
    assert not c['paper_baseline_reproduced'] and not c['selected_checkpoint_step_certified']


def test_native_W2_and_matched_W1_direction_and_truth_identity_are_not_interchanged():
    c=s.registration();profile=s.read(s.ROOT/s.MODEL)
    assert c['native_EMD_kind']=='W2=sqrt(exactOT_squaredEuclidean)'
    assert c['matched_metric']=='W1=exactOT_rawEuclidean'
    assert c['fixed_primary_direction']=='from_past_observed_timepoint_to_leftout_timepoint'
    assert c['matched_reference_role']=='data_identity_only_not_leaveout3_model_performance'
    assert profile['paper_Base_D_leave3_target']==.811
    assert profile['paper_Base_DV_mean_over_three_leaveouts']==.848
    assert not profile['paper_target_variant_identity_certified']


def test_original_metric_early_exit_is_real_and_unchanged_source_functions_are_used():
    c=s.registration();root=s.ROOT/c['source_root']
    tree=ast.parse((root/'TrajectoryNet/eval.py').read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    exits=[n for n in main.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='exit']
    evaluations=[n for n in ast.walk(main) if isinstance(n,ast.Call) and ast.unparse(n.func)=='eval_utils.evaluate_kantorovich_v2']
    assert len(exits)==1 and len(evaluations)==1 and exits[0].lineno<evaluations[0].lineno
    caller=(s.ROOT/(s.MODULE.replace('.','/')+'.py')).read_text()
    assert 'evaluation.integrate_backwards(end,model' in caller
    assert 'native_eval.evaluate_kantorovich_v2(' in caller
    parent=s.read(s.ROOT/s.CFM_REFERENCE)
    for key in ['settings','resource_lock','numerical_threads']:
        assert c[key]==parent[key]
    assert c['settings']['minimum_commit_headroom_bytes']==32*2**30
