import copy
import json
from pathlib import Path

import pytest

from scripts.flow_matching import run_grasp_completion_replay_queue_v1 as queue


def original():
    return dict(id='original', profile='tau_8', dataset='cicids', entity='cicids_7', seed=1112,
        original_id='original_v2', model_config='configured.json', model_config_sha256='pin',
        expected_optimizer_updates=121500, test_points=180569, validation_points=5157,
        score_recipes=[dict(source_samples=5, flow_evaluations=10)], output_directory='native-training')


def result(job):
    return dict(status='completed', job=job, diagnostic=False, epochs_completed=1500,
        optimizer_updates=job['expected_optimizer_updates'], scored_points=job['test_points'],
        validation_points=job['validation_points'], score_recipes=job['score_recipes'],
        metrics=[dict(recipe=r) for r in job['score_recipes']],
        paper_equivalence_certified=False, strict_TAB_result=False)


@pytest.mark.parametrize('field,value', [('seed',1103), ('profile','mean'), ('dataset','swan'),
    ('expected_optimizer_updates',93000), ('test_points',72000), ('validation_points',3935),
    ('model_config_sha256','changed'), ('score_recipes',[]), ('original_id','other')])
def test_recovery_cannot_change_any_scientific_field(field, value):
    job = original()
    execution = dict(job, id='recovered', output_directory='preserved-recovery', recovery_original_job_id=job['id'])
    queue.check_binding(job, execution)
    execution[field] = value
    with pytest.raises(ValueError):
        queue.check_binding(job, execution)


@pytest.mark.parametrize('field,value', [('epochs_completed',1499), ('optimizer_updates',93000),
    ('scored_points',72000), ('validation_points',3935), ('score_recipes',[]), ('metrics',[]),
    ('status','partial'), ('diagnostic',True), ('paper_equivalence_certified',True), ('strict_TAB_result',True)])
def test_short_original_training_or_missing_full_recipe_cannot_enter_replay(field, value):
    job = original()
    completed = result(job)
    queue.check_complete_result(job, job, completed)
    completed[field] = value
    with pytest.raises(ValueError):
        queue.check_complete_result(job, job, completed)


@pytest.mark.parametrize('failure', [None, 'missing_job', 'duplicate_job', 'short_seeds', 'short_cohorts', 'short_scope'])
def test_all_original_9120_jobs_480_cohorts_and_ten_seeds_required(failure):
    source = Path(__file__).resolve().parents[1]/'configs/experiments/fm_grasp_protocol_queue.v4.json'
    registered = json.loads(source.read_text(encoding='utf-8'))
    runtime = dict(required_original_models=9120, required_original_cohorts=480,
        required_original_training_seeds=10, original_scope=copy.deepcopy(queue.SCOPE))
    if failure == 'missing_job': registered['jobs'].pop()
    elif failure == 'duplicate_job': registered['jobs'][-1] = registered['jobs'][0]
    elif failure == 'short_seeds': registered['seeds'] = [1103,1104]
    elif failure == 'short_cohorts': runtime['required_original_cohorts'] = 479
    elif failure == 'short_scope': runtime['original_scope']['papers'] = 44
    if failure:
        with pytest.raises(ValueError): queue.validate_scope(registered, runtime)
    else:
        assert len(queue.validate_scope(registered, runtime)) == 9120


def test_registration_retains_entity_specific_budgets_all_recipes_and_all_native_guards(tmp_path, monkeypatch):
    job = original()
    job['score_recipes'].append(dict(source_samples=10, flow_evaluations=100))
    target = tmp_path/job['output_directory']/'result.json'
    target.parent.mkdir()
    target.write_text(json.dumps(result(job)), encoding='utf-8')
    (tmp_path/'runtime.json').write_text('{}', encoding='utf-8')
    template = dict(settings={'minimum_commit_headroom_bytes':22*2**30}, resource_lock='original-lock',
        python='original-python', cpu_threads=2, score_tolerance={'rtol':.0002,'atol':.001},
        metric_tolerance={'metric_atol':.0002}, original_scope=queue.SCOPE, source_receipts=[], queue='all-original-jobs')
    runtime = dict(source_receipts=[], proof_root='proofs', child_state_root='states', artifact_root='external-data',
        child_configuration_root='configs')
    monkeypatch.setattr(queue, 'ROOT', tmp_path)
    path = queue.registration(runtime, 'runtime.json', template, job, job)
    pipeline = json.loads((tmp_path/path).read_text(encoding='utf-8'))
    native = json.loads((tmp_path/pipeline['producer_configuration']).read_text(encoding='utf-8'))
    assert native['full_score_recipe_count'] == 2 and native['full_job_count'] == 1
    assert native['full_results'][0]['canonical_id'] == job['id']
    for field in queue.GUARDS:
        assert pipeline[field] == native[field] == template[field]
    assert native['queue'] == 'all-original-jobs'
    with pytest.raises(FileExistsError): queue.registration(runtime, 'runtime.json', template, job, job)


def test_queue_parent_and_generic_pipeline_keep_native_training_out_of_the_parent():
    import ast
    tree = ast.parse(Path(queue.__file__).read_text(encoding='utf-8'))
    imports = [n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert all('numpy' not in ast.unparse(n) and 'torch' not in ast.unparse(n) for n in imports)
    text = ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='controller'))
    assert 'run_grasp_registered_replay_review_pipeline_v2' in text
    assert 'grasp_protocol_runner' not in text
