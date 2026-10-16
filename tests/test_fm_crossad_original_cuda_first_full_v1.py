import ast
import json
from types import SimpleNamespace
import pytest
from scripts.flow_matching import run_crossad_original_cuda_first_full_v1 as s


def test_source_AST_changes_only_three_device_fields_and_two_score_placements():
    source=s.ROOT/'scripts/flow_matching/run_crossad_author_job.py'
    text=source.read_text();tree=ast.parse(text)
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['run','native_scores']]
    ns=dict(__name__='_literal_original_CUDA_AST_fixture')
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),ns)
    run,scores=s.configured_functions(SimpleNamespace(**ns))
    assert run.__code__.co_filename.endswith('::original_CUDA_run')
    assert scores.__code__.co_filename.endswith('::native_score_device')
    assert run.__code__.co_names==ns['run'].__code__.co_names
    assert 'cuda' in run.__code__.co_consts and 'cuda:0' in run.__code__.co_consts
    assert all(name in run.__code__.co_names for name in ['train','test','evaluate_spot','evaluate_saved'])
    extra={'next','parameters','device'}
    assert set(scores.__code__.co_names)-set(ns['native_scores'].__code__.co_names)==extra
    assert tuple(name for name in scores.__code__.co_names if name not in extra)==ns['native_scores'].__code__.co_names


def test_full20epoch_budget_existing_seed_and_separate_output_are_bound():
    c=s.registration();base=s.read(s.ROOT/s.preflight.CONFIG)
    assert c['job']['seed']==1105 and c['actual_original_epochs_max']==20
    assert c['job']['train_parameters']==base['job']['train_parameters']
    assert c['job']['model_parameters']==base['job']['model_parameters']
    assert c['job']['output_directory']!=base['job']['output_directory']
    assert c['job']['id']!=base['job']['id'] and c['canonical_CPU_condition_reference']==base['job']['id']
    assert c['host_guards']==base['settings'] and c['GPU']==base['GPU']
    assert c['new_independent_training_seed_slots']==0 and not c['other_original_conditions_removed']
    assert c['settings']['runtime_root']==base['original_source_root']
    assert not c['job']['activation_checkpointing']


def test_all_metric_coverage_and_uncertainty_fields_must_replay():
    tolerance=dict(rtol=2e-6,atol=2e-6)
    value=dict(strict=dict(micro=dict(f1=.5),f1_ci95=dict(low=.4,high=.6)),coverage=[128,128,13])
    s.compare_nested(value,json.loads(json.dumps(value)),tolerance)
    wrong=json.loads(json.dumps(value));wrong['strict']['f1_ci95']['low']=.1
    with pytest.raises(ValueError):s.compare_nested(value,wrong,tolerance)
    wrong=json.loads(json.dumps(value));wrong['coverage']=wrong['coverage'][:-1]
    with pytest.raises(ValueError):s.compare_nested(value,wrong,tolerance)
