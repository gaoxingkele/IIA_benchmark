import copy
import pytest
from scripts.flow_matching import run_cfm_uncovered_resource_aborts_v1 as current

CONFIG='configs/runtime/fm_cfm_all8_uncovered_resource_abort_recovery.v1.json'


def test_every_original_scientific_slot_retained_and_all_eight_uncovered_failures_selected():
    runtime=current.original.read(current.ROOT/CONFIG)
    selected,queue=current.verify(runtime)
    assert len(queue['jobs'])==285 and len(selected)==8
    canonical=current.original.read(current.ROOT/runtime['controllers']['original']['queue'])
    old={j['id']:j for j in canonical['jobs']}
    excluded={'id','recovery_original_job_id','output_directory','artifact_directory'}
    for job in selected:
        original=old[job['recovery_original_job_id']]
        assert {k:v for k,v in original.items() if k not in excluded}=={k:v for k,v in job.items() if k not in excluded}
        assert job['output_directory']!=original['output_directory']


def test_numerical_step_underflow_slot_is_not_automatically_retrained():
    selected,queue=current.verify(current.original.read(current.ROOT/CONFIG))
    assert all(j['recovery_original_job_id']!='cfm_ts_original__appendix_d__bb_gaussian_consistent_e600__sine__seed1107' for j in selected)


def test_lowered_retry_admission_rejected():
    runtime=copy.deepcopy(current.original.read(current.ROOT/CONFIG))
    runtime['retry_admission_commit_headroom_bytes']=20*2**30
    with pytest.raises(ValueError,match='40GiB'):
        current.verify(runtime)


def test_partial_epoch_progress_does_not_reduce_original_optimizer_budget():
    selected,_=current.verify(current.original.read(current.ROOT/CONFIG))
    sine=[j for j in selected if j['dataset']=='sine']
    assert len(sine)==3 and all(j['epochs']==400 and j['expected_optimizer_updates']==25200 for j in sine)
    node=[j for j in selected if j['method']=='node']
    assert len(node)==5 and all(j['expected_optimizer_updates']==j['epochs'] for j in node)
