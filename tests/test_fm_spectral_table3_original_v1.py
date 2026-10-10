import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.flow_matching import run_spectral_table3_original_v1 as original
from scripts.flow_matching import run_spectral_table3_queue_v1 as queue_runner
from scripts.flow_matching.spectral_table3_bootstrap_v1 import check_evaluator_repeats,verify_training_coverage,direct_generator_frame
from scripts.flow_matching.spectral_table2_bootstrap_v1 import execute_original_entry


def test_full_epoch_and_batch_budget_cannot_be_replaced_by_partial_training():
    contract=dict(epochs=1000,batches_per_epoch=3,periodic_evaluations=10)
    state=dict(optimizer_updates=3000,epoch_updates={str(i):3 for i in range(1000)},evaluations=[{}]*10)
    verify_training_coverage(state,contract)
    state['epoch_updates']['999']=2
    with pytest.raises(ValueError,match='epoch/batch'):verify_training_coverage(state,contract)
    state['optimizer_updates']=2999
    with pytest.raises(ValueError,match='update budget'):verify_training_coverage(state,contract)


def test_original_evaluator_ten_repeats_and_population_std_are_required():
    repeats=dict(classification=list(range(10)),predictive=[.2]*10,marginal=[.019]*10)
    scores=dict(clf_score_mean=4.5,clf_score_std=float(np.std(range(10))),
        predictive_score_mean=.2,predictive_score_std=0,marginal_score_mean=.019)
    check_evaluator_repeats(repeats,scores)
    repeats['classification']=list(range(9))
    with pytest.raises(ValueError,match='ten full'):check_evaluator_repeats(repeats,scores)


def test_full_tsf_rows_are_preserved_without_paper_length_crop(tmp_path,monkeypatch):
    monkeypatch.setattr(original,'ROOT',tmp_path)
    raw=tmp_path/'nn5_daily.tsf';raw.write_text('@data\na:1,2,3,4\nb:2,4,6,8\n',encoding='cp1252')
    path=tmp_path/'queue.json';path.write_text('{}')
    queue=dict(queue_path='queue.json',datasets=dict(nn5_daily=dict(raw_path='nn5_daily.tsf',shape=[2,4,1],paper_stated_length=3)))
    proof=original.data_audit(queue,tmp_path/'new/preflight/proof.json')
    record=proof['datasets']['nn5_daily']
    assert record['raw_shape']==[2,4,1] and record['released_length_differs_from_paper']
    with np.load(record['arrays']) as arrays:
        assert arrays['raw'].shape==(2,4,1)
        assert np.allclose(arrays['normalized'].mean(axis=1),0,atol=1e-7)
        assert np.allclose(arrays['normalized'].std(axis=1),1,atol=1e-7)
    assert proof['diagnostic_only'] and not proof['formal_benchmark_result']


def test_original_optimizer_assignment_is_executed_without_replacing_math(tmp_path):
    source=tmp_path/'main.py';source.write_text('def main(args):\n    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay)\n')
    factory=lambda parameters,**kwargs:(parameters,kwargs)
    torch=SimpleNamespace(optim=SimpleNamespace(AdamW=factory))
    model=SimpleNamespace(parameters=lambda:['frozen full parameters'])
    args=SimpleNamespace(learning_rate=.0003,weight_decay=.00001)
    value=original.original_optimizer(source,model,args,torch)
    assert value==(['frozen full parameters'],dict(lr=.0003,betas=(.9,.95),weight_decay=.00001))


def test_optimizer_observer_does_not_count_nested_evaluator_updates(tmp_path):
    entry=tmp_path/'main.py';metric=tmp_path/'metric.py'
    namespace=dict(observer=direct_generator_frame,entry=entry,torch_root=tmp_path/'torch')
    exec(compile('def nested():\n    return observer(entry, torch_root)\n',str(metric),'exec'),namespace)
    exec(compile('def main():\n    return observer(entry, torch_root), nested()\n',str(entry),'exec'),namespace)
    direct,nested=namespace['main']()
    assert direct.f_code.co_filename==str(entry) and nested is None


def test_windows_outer_main_is_retained_and_exact_original_source_executes_once(tmp_path):
    import sys
    source=tmp_path/'main.py';source.write_text("calls.append(__name__)\nif __name__ == '__main__':\n    calls.append('full original loop')\n")
    outer=sys.modules['__main__'];calls=[]
    execute_original_entry(source,[],dict(calls=calls))
    assert calls==['__main__','full original loop'] and sys.modules['__main__'] is outer


def test_native_capacity_proof_rejects_reduced_batch_and_worker_counts(tmp_path,monkeypatch):
    monkeypatch.setattr(queue_runner,'ROOT',tmp_path)
    path=tmp_path/'queue.json';path.write_text('{}')
    config=dict(dataset='fred_md',author_hyperparameters=dict(batch_size=32))
    (tmp_path/'model.json').write_text(json.dumps(config))
    job=dict(id='full',model_config='model.json',model_config_sha256=original.sha(tmp_path/'model.json'))
    queue=dict(queue_path='queue.json',datasets=dict(fred_md=dict(shape=[107,728,1])))
    proof=dict(passed=True,diagnostic_only=True,queue_sha256=original.sha(path),job_id='full',
        full_original_config_sha256=job['model_config_sha256'],training_batch_shape=[32,728,1],test_shape=[22,728,1],
        full_loader_rows=85,num_workers=4,worker_pids=[11,12,13,14],worker_exitcodes=[0]*4,
        native_full_generation_and_ten_repeat_evaluator_passed=True)
    assert queue_runner.full_capacity_proof(queue,job,proof)
    proof['training_batch_shape'][0]=2
    assert not queue_runner.full_capacity_proof(queue,job,proof)
    proof['training_batch_shape'][0]=32;proof['worker_pids']=[11,11,12,13]
    assert not queue_runner.full_capacity_proof(queue,job,proof)


def test_missing_or_nonfinite_raw_values_cannot_be_silently_filled(tmp_path):
    p=tmp_path/'raw.tsf';p.write_text('@data\na:1,?,3\n')
    with pytest.raises(ValueError):original.parse_tsf(p)
    p.write_text('@data\na:1,nan,3\n')
    with pytest.raises(ValueError,match='Nonfinite'):original.parse_tsf(p)
def test_environment_freeze_preserves_all_numerical_versions():
    from scripts.flow_matching.freeze_spectral_table3_environment_v1 import verify_versions
    import pytest
    expected = {'pip': '26.2.1', 'torch': '2.8.0+cu126', 'numpy': '1.26.4'}
    verify_versions(expected, dict(expected, pip='23.0.1'), {'pip': '23.0.1'})
    with pytest.raises(ValueError, match='Inherited package differs'):
        verify_versions(expected, dict(expected, numpy='2.0.0'), {})
    with pytest.raises(ValueError, match='Only the explicitly registered pip'):
        verify_versions(expected, expected, {'numpy': '1.26.4'})


