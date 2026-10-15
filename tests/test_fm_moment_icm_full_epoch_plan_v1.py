from collections import Counter
import copy
import pytest
from scripts.flow_matching import prepare_moment_icm_full_epoch_plan_v1 as p


def test_complete_all_four_branch_scope_includes_all322_pending_sources_without_readiness_claim():
    c=p.registration();sources=p.verify(c);counts=c['count_audit']
    assert len(sources)==2086 and len(c['excluded'])==76
    assert Counter(s['family'] for s in sources)==dict(classification=95,long_forecast=9,monash=33,anomaly_original=1627,anomaly_finite_control=322)
    assert all(not s['cache_independent_closure_complete'] for s in sources if s['family']=='anomaly_finite_control')
    assert counts['families']['anomaly_original']['items']==72417674
    assert counts['optimizer_batch_slots_per_full_epoch']==sum(b['batches'] for b in counts['width_buckets'])
    assert counts['full_tensor_time_positions_including_overlap_and_repeated_remainder_channels']==counts['full_patch_token_positions_including_padding']*8


@pytest.mark.parametrize('mutation',['missing322','shortened_epoch','drop_last','changed_batch','other_monash','other_case_split','relaxed_guard'])
def test_no_partial_scope_shortened_budget_or_other_recipe_can_masquerade_as_full_epoch(mutation):
    c=p.registration()
    if mutation=='missing322':c['sources']=[s for s in c['sources'] if s['family']!='anomaly_finite_control']
    elif mutation=='shortened_epoch':c['data_source_fraction']=.1
    elif mutation=='drop_last':c['drop_last']=True
    elif mutation=='changed_batch':c['batch_size']=64
    elif mutation=='other_monash':c['monash_selection']='temporal_control'
    elif mutation=='other_case_split':c['classification_train_selection']='flattened'
    else:c['settings']['minimum_commit_headroom_bytes']=1
    with pytest.raises(ValueError):p.verify(c)


def test_complete_worker_body_writes_full_fixture_plan_and_bijection_proof_in_native_environment(tmp_path):
    import os,subprocess
    code=r'''
from pathlib import Path
import sys,json
from scripts.flow_matching import prepare_moment_icm_full_epoch_plan_v1 as p
original=p.registration();p.ROOT=Path(sys.argv[1])
sources=[dict(dataset_key='fixture1',width=1,split='train',items=101),dict(dataset_key='fixture8',width=8,split='train',items=65)]
config=dict(environment=original['environment'],sources=sources,batch_size=32,epoch_seed=13,
 artifact_root=str(p.ROOT/'artifact'),count_audit=dict(total_items=166))
p.write(p.ROOT/p.CONFIG,config);p.verify=lambda c:sources
p.worker(config)
proof=p.read(p.ROOT/'artifact'/p.REPORT)
assert proof['passed'] and proof['index_closure']['complete_epoch_items']==166
assert proof['index_closure']['all_original_source_items_exactly_once']
assert not proof['complete_training_data_ready'] and not proof['actual_model_or_data_window_materialization_executed']
assert 'torch' not in sys.modules
'''
    result=subprocess.run([str(p.ROOT/'.venv/moment-author-py310/Scripts/python.exe'),'-B','-X','utf8','-c',code,str(tmp_path)],cwd=p.ROOT,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout[-1000:]+result.stderr[-4000:]
