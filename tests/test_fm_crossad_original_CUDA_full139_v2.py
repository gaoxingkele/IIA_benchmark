from scripts.flow_matching import run_crossad_original_CUDA_full139_v2 as s


def test_every139_condition_and_all_numerical_gates_are_preserved_in_isolated_paths():
    c=s.registration();p=s.read(s.ROOT/s.parent.CONFIG)
    assert len(c['jobs'])==len(p['jobs'])==139
    for old,new in zip(p['jobs'],c['jobs']):
        for key in ['canonical_CPU_condition_reference','dataset','seed','model_parameters','train_parameters',
                    'raw_sha256','train_length','frozen_files','mode','reconstructed_ablation']:
            assert old[key]==new[key]
        assert old['id']!=new['id'] and old['output_directory']!=new['output_directory']
        if old['reconstructed_ablation']:assert old['ablation_row']==new['ablation_row']
    assert c['GPU']==p['GPU'] and c['host_guards']==p['host_guards'] and c['array_tolerance']==p['array_tolerance']
    assert c['delegated_first_configuration']==s.recovery.CONFIG
    assert c['completed20_training_reused_without_additional_updates'] and c['original_failed_full139_attempt_preserved']


def test_controller_numeric_dispatch_and_dependency_logic_are_unchanged(monkeypatch):
    controller=s.controller_callable()
    assert controller.__code__ is s.parent.controller.__code__
    assert controller.__globals__['verify'] is s.verify and controller.__globals__['MODULE']==s.MODULE
    assert controller.__globals__['first_completion'] is s.parent.first_completion
    seen=[]
    monkeypatch.setattr(s.parent,'native_probe',lambda c,j,cpu:seen.append(('native',cpu)))
    monkeypatch.setattr(s.parent,'source_or_replay',lambda c,j,replay:seen.append(('source',replay)))
    monkeypatch.setattr(s.recovery,'install_evaluator',lambda c:seen.append(('evaluator',None)))
    for phase in s.parent.PHASES:s.dispatch({}, {}, phase)
    assert seen==[('native',False),('native',True),('evaluator',None),('source',False),('evaluator',None),('source',True)]
