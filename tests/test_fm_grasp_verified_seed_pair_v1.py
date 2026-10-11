from copy import deepcopy
import pytest
from scripts.flow_matching.audit_grasp_verified_seed_pair_v1 import verify_seed_bindings,verify_actual_execution
from scripts.flow_matching.audit_grasp_checkpoint_replay_v1 import SCIENTIFIC_FIELDS


def test_recovery_execution_alias_preserves_original_scientific_identity_and_seed():
    job={k:k for k in SCIENTIFIC_FIELDS};job.update(id='canonical',seed=1103)
    recovered=dict(job,id='canonical__verified_recovery',output_directory='preserved_new_attempt')
    result=dict(job=recovered)
    assert verify_actual_execution(job,result,dict(jobs=[recovered]))==recovered
    for key in SCIENTIFIC_FIELDS:
        wrong=deepcopy(result);wrong['job'][key]='changed'
        with pytest.raises(ValueError):verify_actual_execution(job,wrong,dict(jobs=[wrong['job']]))
    with pytest.raises(ValueError):verify_actual_execution(job,result,dict(jobs=[job]))


def config():
    return dict(profile='tau_0p25',completed_seed_bindings={str(s):dict(
        canonical_id=f'grasp_protocol_v3__tau_0p25__swan__swan_sf__seed{s}',result_path=f'actual_seed{s}_result.json')
        for s in (1103,1104)})


def test_actual_recovery_cannot_inflate_independent_seed_count_or_replace_other_seed():
    verify_seed_bindings(config())
    for kind in ['duplicate_result','relabeled_seed','missing_seed','extra_retry']:
        value=deepcopy(config())
        if kind=='duplicate_result':value['completed_seed_bindings']['1104']['result_path']=value['completed_seed_bindings']['1103']['result_path']
        if kind=='relabeled_seed':value['completed_seed_bindings']['1104']['canonical_id']=value['completed_seed_bindings']['1103']['canonical_id']
        if kind=='missing_seed':del value['completed_seed_bindings']['1104']
        if kind=='extra_retry':value['completed_seed_bindings']['retry']=value['completed_seed_bindings']['1103']
        with pytest.raises(ValueError):verify_seed_bindings(value)
