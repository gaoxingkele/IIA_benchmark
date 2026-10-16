import ast
import json
import subprocess
import sys
import pytest
from scripts.flow_matching import recover_crossad_original_CUDA_completed20_v1 as s


def test_exact_completed20_actual_trace_checkpoint_and_terminal_import_failure_are_bound():
    c=s.registration();s.verify(c)
    trace=s.read(s.Path(c['original_training_output'])/'training_trace.json')
    assert trace['epochs_finished']==20 and trace['training_forward_calls']==8640
    assert c['job']['train_parameters']['train_epochs']==20
    assert c['additional_training_updates_in_recovery']==0 and c['source_training_seconds_unknown']
    previous=s.read(s.ROOT/s.first.CONFIG)
    assert c['job']['output_directory']!=previous['job']['output_directory']
    assert c['artifact_root']!=previous['artifact_root']


def test_only_finished_training_is_restored_while_all_source_evaluations_remain():
    path=s.ROOT/'scripts/flow_matching/run_crossad_author_job.py'
    tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    namespace=dict(__name__='_actual_source_recovery_AST_contract')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    run=s.evaluation_only_function(namespace['run'],namespace,lambda exp:None,
        dict(training_forward_calls=8640,validation_losses=[.5]*20))
    assert 'restore_completed_training' in run.__code__.co_names
    assert 'restored_training_accounting' in run.__code__.co_names
    assert all(name in run.__code__.co_names for name in ['test','evaluate_spot','evaluate_saved','native_scores'])
    assert 'cuda' in run.__code__.co_consts and 'cuda:0' in run.__code__.co_consts
    assert run.__code__.co_filename.endswith('::continue_after_completed20')


def test_original_real_forward_accounting_cannot_append_fake_training():
    count=s.CompletedForwardCount(8640)
    assert len(count)==8640
    with pytest.raises(ValueError):count.append(.5)


def test_actual_optional_evaluator_import_passes_without_touching_original_GPU_model():
    code="from scripts.flow_matching import recover_crossad_original_CUDA_completed20_v1 as s;import json,sys; c=s.registration();r=s.install_evaluator(c);r['PATE_loaded']=any(n.startswith('ts_ad_evaluation.pate') for n in sys.modules);print(json.dumps(r))"
    result=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,text=True,encoding='utf-8',capture_output=True,check=True)
    report=json.loads(result.stdout.strip().splitlines()[-1])
    assert report['all_requested_metric_implementations_unchanged'] and not report['PATE_loaded']
    assert report['requested_author_metrics']==['best_f1','auc','r_auc','vus']
    model=s.ROOT/s.read(s.ROOT/s.first.preflight.CONFIG)['original_source_root']/'models/CrossAD/Basic_CrossAD.py'
    assert model.read_text().count('.cuda()')==2
