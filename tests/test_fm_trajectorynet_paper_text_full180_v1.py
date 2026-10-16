from scripts.flow_matching import run_trajectorynet_EB_paper_text_full180_v1 as s
from scripts.flow_matching import run_trajectorynet_EB_native_full180_v1 as original


def test_paper_recipe_is_separate_and_changes_only_two_declared_source_parameters():
    source=original.registration();paper=s.driver.registration()
    changed={k for k in source['source_parameters'] if source['source_parameters'][k]!=paper['source_parameters'][k]}
    assert changed=={'nonlinearity','weight_decay'}
    assert source['source_parameters']['nonlinearity']=='tanh' and source['source_parameters']['weight_decay']==1e-5
    assert paper['source_parameters']['nonlinearity']==paper['activation_extension']['key']
    assert paper['source_parameters']['weight_decay']==5e-5
    assert paper['activation_extension']['negative_slope']==.01
    assert not paper['activation_extension']['author_negative_slope_known']
    assert not paper['activation_extension']['original_source_activation_table_has_LeakyReLU']
    assert original.MODULE=='scripts.flow_matching.run_trajectorynet_EB_native_full180_v1'
    assert {j['id'] for j in source['jobs']}.isdisjoint({j['id'] for j in paper['jobs']})
    assert {j['artifact_directory'] for j in source['jobs']}.isdisjoint({j['artifact_directory'] for j in paper['jobs']})


def test_complete_grid_native_math_and_guards_are_reused_without_shortening():
    source=original.registration();paper=s.driver.registration()
    assert len(paper['jobs'])==135 and len(paper['deferred_energy_jobs'])==45
    assert [(j['recipe'],j['seed'],j['leftout_timepoint']) for j in paper['jobs']]==[(j['recipe'],j['seed'],j['leftout_timepoint']) for j in source['jobs']]
    for key in ['settings','resource_lock','actual_optimizer_steps_per_training_model','batch_size',
        'checkpoint_selections','native_probe_recipe_leaveout_cases','metric_tolerance','require_bitwise_full_prediction_replay']:
        assert paper[key]==source[key]
    for name in ['train','replay','probe','metrics','aggregate','controller','verify']:
        a,b=getattr(s.driver,name).__code__,getattr(original,name).__code__
        assert a.co_code==b.co_code and a.co_consts==b.co_consts


def test_only_the_actually_included_source_table_defines_published_targets():
    audit=s.read(s.ROOT/'projects/flow_matching_research/ara/adopted_baselines/trajectorynet_paper_text_vs_logged_recipe.2026-10-16_v1.json')
    assert audit['active_Table3_include_chain'][-1]=='tables/eb_table_v5.tex'
    assert audit['active_Table3_all40_scalars_match_pdf_audit']
    assert audit['active_Table3']['Base+D+V'][-1]==.848 and audit['active_Table3']['Base+E'][-1]==.825
    assert audit['published_recipe']['activation']=='LeakyReLU'
    assert audit['released_author_log_recipe']['activation']=='Tanh'
    assert not audit['energy_recipe']['EB_specific_Jacobian_weight_or_selected_grid_cell_found']
    assert not audit['source_execution_performed']
