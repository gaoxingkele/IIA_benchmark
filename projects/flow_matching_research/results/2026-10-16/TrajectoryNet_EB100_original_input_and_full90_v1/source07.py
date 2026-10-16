import ast
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.flow_matching import run_otcfm_EB100_code_referenced_native90_v1 as s


def test_every_original_EB100_case_preserves_three_leaveouts_five_seeds_and_both_full_recipes():
    cases=s.scientific_cases()
    assert len(cases)==len({j['id'] for j in cases})==90
    assert {j['dataset'] for j in cases}=={'trajectorynet_EB100_code_referenced'}
    assert {j['seed'] for j in cases}=={42,43,44,45,46}
    assert {j['leaveout_timepoint'] for j in cases}=={1,2,3}
    assert {j['method'] for j in cases}=={'cfm','otcfm','sbcfm'}
    assert all(j['scientific_protocol']['max_epochs']==1000 for j in cases)
    assert len(s.preflight_cases(dict(jobs=cases)))==18


@pytest.mark.parametrize('fault',['missing_leave3','missing_seed42_method','duplicate','changed_recipe'])
def test_preflight_cannot_narrow_any_original_data_or_method_branch(fault):
    cases=deepcopy(s.scientific_cases())
    if fault=='missing_leave3':cases=[j for j in cases if j['leaveout_timepoint']!=3]
    elif fault=='missing_seed42_method':cases=[j for j in cases if not(j['seed']==42 and j['method']=='sbcfm')]
    elif fault=='duplicate':cases.append(deepcopy(cases[0]))
    else:cases[0]['protocol']='shortened_proxy'
    with pytest.raises(ValueError):s.preflight_cases(dict(jobs=cases))


def test_new_original_input_scope_preserves_all_prior270_and_every_guard():
    q=s.registration();old=s.read(s.ROOT/'configs/experiments/fm_otcfm_EB_CITE_native_full150.v2.json');base=s.read(s.ROOT/s.original.QUEUE)
    ids=[j['id'] for queue in [q,old,base] for j in queue['jobs']]
    assert len(ids)==len(set(ids))==360
    assert q['settings']==old['settings']==base['settings'] and q['resource_lock']==base['resource_lock']
    assert {j['data']['cells'] for j in q['jobs']}=={16819}
    assert {j['data']['full_original_PC_dimensions'] for j in q['jobs']}=={100}
    for j in q['jobs']:
        plan=j['validation_compatibility'];batch=j['scientific_protocol']['batch_size']
        assert plan['original_training_batch']==batch
        assert plan['effective_validation_batch']==(238 if batch==512 else 128)
        assert j['data']['original_code_referenced_public_asset_verified']


def test_full_native_mathematical_worker_is_identical_to_already_executed150_body():
    root=Path(__file__).resolve().parents[1]
    def body(path):
        return ast.dump(next(n for n in ast.parse((root/path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='worker'),include_attributes=False)
    assert body('scripts/flow_matching/run_otcfm_EB100_code_referenced_native90_v1.py')==body('scripts/flow_matching/run_otcfm_EB_CITE_native150_v2.py')


def test_exact_original_first_timepoint_fraction_allocation_preserves_all2381_cells():
    assert s.contract.fraction_lengths(2381,[.8,.1,.1])==[1905,238,238]
