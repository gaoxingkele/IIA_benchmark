import ast
import copy
import inspect
import json
from pathlib import Path

import pytest

from scripts.flow_matching import run_spectral_energy42_preserved_full_retry_v1 as retry


def fixture():
    path=Path(__file__).resolve().parents[1]/'configs/experiments/fm_spectral_original_resource_admission_queue.v3.json'
    original=json.loads(path.read_text());queue=copy.deepcopy(original)
    config=dict(state_root='new_state',selected_output_directory='new_output',selected_artifact_directory='new_artifact')
    queue['state_root']=config['state_root'];job=next(j for j in queue['jobs'] if j['id']==retry.IDENTITY)
    job.update(output_directory='new_output',artifact_directory='new_artifact')
    return original,queue,config,job


def test_all70_jobs_and_full_failed_recipe_preserved():
    original,queue,config,job=fixture();old,new=retry.exact_retry(original,queue,config)
    assert old['training_updates']==new['training_updates']==25000
    assert new['data_shape']==[19712,24,28] and new['generation_batch_size']==2001
    assert len(queue['jobs'])==70 and old['seed']==new['seed']==42


@pytest.mark.parametrize('change',['missing','seed','updates','batch','shape','chunk','guard','other','output','artifact'])
def test_narrowed_scope_changed_science_or_overwrites_are_rejected(change):
    original,queue,config,job=fixture()
    if change=='missing':queue['jobs'].pop()
    elif change=='seed':job['seed']=1103
    elif change=='updates':job['training_updates']=1
    elif change=='batch':job['generation_batch_size']=1
    elif change=='shape':job['data_shape']=[100,24,28]
    elif change=='chunk':job['generation_field_chunk_size']=1
    elif change=='guard':queue['settings']['emergency_gpu_free_bytes']=0
    elif change=='other':queue['jobs'][0]['training_updates']=1
    elif change=='output':job['output_directory']=next(j for j in original['jobs'] if j['id']==retry.IDENTITY)['output_directory']
    elif change=='artifact':job['artifact_directory']=next(j for j in original['jobs'] if j['id']==retry.IDENTITY)['artifact_directory']
    with pytest.raises(ValueError):retry.exact_retry(original,queue,config)


def test_dispatch_ast_only_selects_one_exact_full_failed_slot():
    source=inspect.getsource(retry.ordering.previous.original.main)
    base=retry.ordering.ordered_main_ast(source);expected=copy.deepcopy(base)
    loop=next(n for n in ast.walk(expected) if isinstance(n,ast.For) and ast.unparse(n.iter)=='scheduler_jobs(queue)')
    loop.iter=ast.parse("[j for j in scheduler_jobs(queue) if j['id']==selected_canonical_id]",mode='eval').body
    assert ast.dump(retry.selected_ast(source))==ast.dump(expected)


def proof_fixture():
    proof=dict(passed=True,full_native_Trainer_sample_executed_twice=True,original_generated_shape=[4002,24,6],
        reviewed_generated_shape=[4002,24,6],complete_field_calls_per_pass=1000,
        every_full_field_input_derivative_time_and_auxiliary_bitwise_equal=True,
        all4002_generated_windows_bitwise_equal=True,final_Torch_CPU_RNG_state_equal=True)
    receipt=dict(exit_code=0,resource_guard_aborted=False,full_native_Trainer_sampling_graph_audit_passed=True)
    return proof,receipt


def test_actual_shared_component_proof_accepted_without_energy_or_cuda_certificate():
    proof,receipt=proof_fixture();retry.prerequisite(proof,receipt)
    assert 'Energy_capacity_certified' not in proof


@pytest.mark.parametrize('change',['failed','abort','false','shape','calls','gradient','rng'])
def test_partial_or_failed_shared_component_does_not_launch_training(change):
    proof,receipt=proof_fixture()
    if change=='failed':receipt['exit_code']=1
    elif change=='abort':receipt['resource_guard_aborted']=True
    elif change=='false':proof['passed']=False
    elif change=='shape':proof['original_generated_shape']=[2,24,6]
    elif change=='calls':proof['complete_field_calls_per_pass']=1
    elif change=='gradient':proof['every_full_field_input_derivative_time_and_auxiliary_bitwise_equal']=False
    elif change=='rng':proof['final_Torch_CPU_RNG_state_equal']=False
    with pytest.raises(ValueError):retry.prerequisite(proof,receipt)
