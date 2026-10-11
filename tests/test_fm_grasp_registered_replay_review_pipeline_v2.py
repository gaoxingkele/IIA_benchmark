import copy
import json
from pathlib import Path

import pytest

from scripts.flow_matching import run_grasp_registered_replay_review_pipeline_v2 as m


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


def fixture(tmp_path,monkeypatch):
    from scripts.flow_matching import audit_grasp_checkpoint_replay_v1 as original
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(original,'ROOT',tmp_path)
    producer=dict(settings={'minimum_commit_headroom_bytes':22*2**30},resource_lock='same-lock',python='same-python',cpu_threads=2,
        score_tolerance=dict(rtol=.0002,atol=.001),metric_tolerance=dict(metric_atol=.0002,threshold_rtol=.0002,
            threshold_atol=.001,maximum_prediction_disagreement_rate=.0001),full_job_count=1,full_score_recipe_count=1,
        full_results=[dict(canonical_id='tau1_seed1104',result_path='result.json',result_sha256='hash')],
        output_directory='native-output',state_root='native-state',source_receipts=[],queue='full-queue')
    save(tmp_path/'producer.json',producer)
    config={k:copy.deepcopy(producer[k]) for k in ['settings','resource_lock','python','cpu_threads','score_tolerance','metric_tolerance']}
    config.update(canonical_id='tau1_seed1104',producer_configuration='producer.json',source_receipts=[],
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        review_output_directory='reviewed')
    result=dict(status='completed',epochs_completed=1500,optimizer_updates=93000,
        job={'expected_optimizer_updates':93000},parameter_count=82,metrics=[{'recipe':dict(source_samples=5,flow_evaluations=10)}])
    save(tmp_path/'result.json',result)
    record=dict(canonical_id='tau1_seed1104',epochs_completed=1500,optimizer_updates=93000,parameter_count=82,
        full_checkpoint_restoration_and_reinference_passed=True,result_sha256='hash',
        recipes=[dict(recipe=dict(source_samples=5,flow_evaluations=10))])
    report=dict(configuration_sha256=m.sha(tmp_path/'producer.json'),score_tolerance=producer['score_tolerance'],
        metric_tolerance=producer['metric_tolerance'],full_job_count=1,
        full_checkpoint_contents_and_re_inference_passed=True,additional_training_seed_slots=0,
        cuda_context_initialized=False,records=[record])
    receipt=dict(exit_code=0,resource_guard_aborted=False,full_replay_passed=True)
    save(tmp_path/'native-output/checkpoint_replay_audit.json',report)
    save(tmp_path/'native-output/tau1_seed1104.json',dict(completed_recipes=record['recipes']))
    save(tmp_path/'native-state/receipt.json',receipt)
    for source in ['verify_grasp_completed_full_replay_v1.py','audit_grasp_checkpoint_replay_resume_v2.py']:
        target=tmp_path/'scripts/flow_matching'/source
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((Path(__file__).resolve().parents[1]/'scripts/flow_matching'/source).read_bytes())
    monkeypatch.setattr(original,'verify_registration',lambda _:dict(jobs=[None]*9120,cohort_count=480))
    return config,producer,report,receipt


@pytest.mark.parametrize('changed',[None,'settings','resource_lock','python','cpu_threads','score_tolerance','metric_tolerance','canonical_id','scope'])
def test_original_full_matrix_and_all_native_safeguards_required(tmp_path,monkeypatch,changed):
    config,producer,_,_=fixture(tmp_path,monkeypatch)
    if changed is None:assert m.binding(config)==producer;return
    if changed=='scope':config['original_scope']['papers']=44
    elif changed in ['score_tolerance','metric_tolerance']:config[changed]={}
    else:config[changed]='changed'
    with pytest.raises(ValueError):m.binding(config)


@pytest.mark.parametrize('changed',[None,'exit','guard','missing_recipe','short_epochs','budget','configuration','score_tolerance','metric_tolerance','duplicate','cuda','new_seed'])
def test_review_requires_actual_full_native_report_and_terminal_receipt(tmp_path,monkeypatch,changed):
    config,producer,report,receipt=fixture(tmp_path,monkeypatch)
    if changed is None:assert m.terminal(config,producer)==(report,receipt);return
    if changed=='exit':receipt['exit_code']=1
    elif changed=='guard':receipt['resource_guard_aborted']=True
    elif changed=='missing_recipe':report['records'][0]['recipes']=[]
    elif changed=='short_epochs':report['records'][0]['epochs_completed']=1499
    elif changed=='budget':report['records'][0]['optimizer_updates']=92999
    elif changed=='configuration':report['configuration_sha256']='different'
    elif changed=='score_tolerance':report['score_tolerance']={}
    elif changed=='metric_tolerance':report['metric_tolerance']={}
    elif changed=='duplicate':report['records']*=2
    elif changed=='cuda':report['cuda_context_initialized']=True
    elif changed=='new_seed':report['additional_training_seed_slots']=1
    save(tmp_path/'native-output/checkpoint_replay_audit.json',report);save(tmp_path/'native-state/receipt.json',receipt)
    with pytest.raises(ValueError):m.terminal(config,producer)


def test_review_registration_reuses_all_complete_native_recipes_without_new_inference(tmp_path,monkeypatch):
    config,producer,report,_=fixture(tmp_path,monkeypatch)
    review=m.review_registration(config,producer,report)
    assert review['pending_full_recipe_count']==0 and len(review['reusable_complete_recipes'])==1
    assert review['full_results']==producer['full_results']
    assert review['reusable_complete_recipes'][0]['recipe']==dict(source_samples=5,flow_evaluations=10)
    for k in ['settings','resource_lock','python','cpu_threads','score_tolerance','metric_tolerance']:
        assert review[k]==producer[k]


def test_parent_stays_lightweight_and_dispatches_the_existing_full_native_worker():
    import ast
    tree=ast.parse(Path(m.__file__).read_text())
    imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert all('numpy' not in ast.unparse(n) and 'torch' not in ast.unparse(n) for n in imports)
    main=ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main'))
    assert 'audit_grasp_checkpoint_replay_v1' in main and 'grasp_protocol_runner' not in main


def test_generic_completion_receipt_records_actual_seed_instead_of_fixed_seed1104(tmp_path, monkeypatch):
    config, producer, _, _ = fixture(tmp_path, monkeypatch)
    result=json.loads((tmp_path/'result.json').read_text(encoding='utf-8'))
    result['job']['seed']=1112
    save(tmp_path/'result.json',result)
    config.update(review_configuration='completed-review.json',output_directory='completion')
    save(tmp_path/'pipeline.json',config)
    def fake_verify(review,path):
        save(tmp_path/'reviewed/independent_full_replay_verification.json',dict(passed=True,
            all_original_forward_counts_and_recipe_budgets_verified=True,
            all_metrics_independently_recomputed_from_full_replay=True,
            all_saved_full_score_arrays_and_identities_reverified=True,actual_completed_recipe_count=1))
    monkeypatch.setattr(m,'independent_verify',fake_verify)
    m.worker(config,'pipeline.json')
    proof=json.loads((tmp_path/'completion/completed_native_and_independent_review.json').read_text(encoding='utf-8'))
    assert proof['original_training_seed']==1112
    assert proof['required_original_cohort_repeats']==10
    assert 'original_seed1104_required_cohort_repeats' not in proof
    assert not proof['full_ten_seed_cohort_completed'] and not proof['all_original_experiments_complete']
