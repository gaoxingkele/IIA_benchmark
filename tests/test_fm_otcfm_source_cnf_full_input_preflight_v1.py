from scripts.flow_matching import preflight_otcfm_source_cnf_full_inputs_v1 as s


def test_all_five_current_full_inputs_and_all12_leftout_branches_are_retained():
    c=s.registration()
    assert len(c['cases'])==12 and len({v['dataset'] for v in c['cases']})==5
    assert c['training_batch']==1024 and c['dimensions']==5 and c['optimizer_updates']==0
    for v in c['cases']:assert 1<=v['leftout_timepoint']<len(v['data']['days'])-1


def test_original_author_environment_and_all_existing_native_guards_remain():
    c=s.registration();base=s.read(s.ROOT/s.QUEUES[-1])
    assert c['environment_lock']==base['environment_lock'] and c['settings']==base['settings']
    assert c['resource_lock']==base['resource_lock'] and c['python']==base['python']
    assert c['settings']['minimum_commit_headroom_bytes']==32*2**30


def test_source_CNFlow_optimizer_and_benchmark_status_are_explicit():
    c=s.registration();profile=s.read(s.ROOT/s.MODEL)
    assert c['optimizer']==dict(profile['optimizer']) and c['optimizer']['weight_decay']==.01
    assert not profile['full_training_complete'] and not profile['historical_paper_equivalence_certified']
    assert c['augmented_solver']==dict(method='dopri5',atol=1e-5,rtol=1e-5,sensitivity='autograd')
