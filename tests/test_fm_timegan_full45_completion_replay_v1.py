"""Full source-loop fixture checks validate replay plumbing, not performance."""
import ast
import copy
import inspect
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, mean_absolute_error

from scripts.flow_matching import run_timegan_full45_completion_replay_v1 as replay


def native_source(kind):
    model = replay.read(replay.ROOT / 'configs/models/fm_timegan_native_original.v1.json')
    return replay.ROOT / model['source_root'] / 'metrics' / (kind + '_metrics.py')


def configuration():
    queue_path = 'configs/experiments/fm_timegan_native_original_queue.v1.json'
    queue = replay.read(replay.ROOT / queue_path)
    model = replay.read(replay.ROOT / queue['model_config'])
    datasets = replay.read(replay.ROOT / model['dataset_config'])['datasets']
    routes = [dict(canonical_job=copy.deepcopy(job), execution_job=copy.deepcopy(job), kind='original',
        full_original_data_shape=datasets[job['dataset']]['full_shape'], recovery_configuration=None,
        native_completion_receipts=[dict(path='fixture/receipt.json', successful_field='full_native_result_verified')],
        result_path=(Path(job['output_directory']) / 'result.json').as_posix()) for job in queue['jobs']]
    windows = sum(r['full_original_data_shape'][0] for r in routes)
    config = dict(routes=routes, settings=copy.deepcopy(queue['settings']), resource_lock=queue['resource_lock'],
        python=queue['python'], cpu_threads=2, required_checkpoints_each=23, required_evaluators_each=20,
        equality_gate='bitwise', protocol_iterations_each=70000, required_full_generated_windows=windows,
        required_full_predictive_test_windows=10*windows,
        required_full_discriminative_test_windows=sum(20 * (r['full_original_data_shape'][0] - int(r['full_original_data_shape'][0]*.8)) for r in routes))
    return queue, config


def test_every_original_job_module_dataset_seed_and_complete_counts_required():
    queue, config = configuration()
    assert len(replay.check_routes(queue, config)) == 45
    assert {(r['canonical_job']['dataset'], r['canonical_job']['module'], r['canonical_job']['seed']) for r in config['routes']} == {
        (d,m,s) for d in ('stock','energy','sine') for m in ('gru','lstm','lstmLN') for s in range(1103,1108)}
    assert config['required_full_generated_windows'] == 500580
    assert config['required_full_predictive_test_windows'] == 5005800
    assert config['required_full_discriminative_test_windows'] == 2002800


@pytest.mark.parametrize('key,value', [
    ('required_checkpoints_each', 22), ('required_evaluators_each', 19), ('protocol_iterations_each', 69999),
    ('equality_gate', 'allclose'), ('required_full_generated_windows', 500579),
    ('required_full_predictive_test_windows', 5005799), ('required_full_discriminative_test_windows', 2002799),
    ('cpu_threads', 1), ('python', 'other_python'), ('resource_lock', 'other_lock')])
def test_shortened_or_changed_full_scope_rejected(key, value):
    queue, config = configuration(); config[key] = value
    with pytest.raises(ValueError): replay.check_routes(queue, config)


@pytest.mark.parametrize('mutation', ['missing_job','duplicate_job','changed_seed','changed_output','short_shape','missing_receipt','unknown_route','looser_guard'])
def test_incomplete_or_wrong_native_routes_rejected(mutation):
    queue, config = configuration()
    if mutation == 'missing_job': config['routes'].pop()
    elif mutation == 'duplicate_job': config['routes'][-1] = copy.deepcopy(config['routes'][0])
    elif mutation == 'changed_seed': config['routes'][0]['canonical_job']['seed'] = 7
    elif mutation == 'changed_output': config['routes'][0]['execution_job']['output_directory'] = 'other'
    elif mutation == 'short_shape': config['routes'][0]['full_original_data_shape'] = [1,24,6]
    elif mutation == 'missing_receipt': config['routes'][0]['native_completion_receipts'] = []
    elif mutation == 'unknown_route': config['routes'][0]['kind'] = 'fabricated'
    elif mutation == 'looser_guard': config['settings']['emergency_commit_headroom_bytes'] -= 1
    with pytest.raises(ValueError): replay.check_routes(queue, config)


@pytest.mark.parametrize('kind,iterations', [('discriminative',2000),('predictive',5000)])
def test_original_full_source_protocol_and_unused_loss_contract(kind, iterations):
    contract = replay.protocol_contract(kind, native_source(kind).read_text(encoding='utf-8'))
    assert contract['iterations'] == iterations and contract['unused_display_loss_proven']


@pytest.mark.parametrize('kind', ['discriminative','predictive'])
def test_loss_used_for_branching_cannot_be_suppressed(kind):
    source = native_source(kind).read_text(encoding='utf-8')
    name = 'step_d_loss' if kind == 'discriminative' else 'step_p_loss'
    source = source.replace('  ## Test', '  if %s > 0: raise ValueError()\n  ## Test' % name)
    with pytest.raises(ValueError): replay.protocol_contract(kind, source)


@pytest.mark.parametrize('kind,old,new', [('discriminative','iterations = 2000','iterations = 1999'),
    ('predictive','iterations = 5000','iterations = 4999')])
def test_shortened_source_protocol_rejected(kind, old, new):
    with pytest.raises(ValueError):
        replay.protocol_contract(kind, native_source(kind).read_text(encoding='utf-8').replace(old,new))


@pytest.mark.parametrize('mutation', ['ulp','dtype','shape','nan','signed_zero'])
def test_predictions_and_variables_require_finite_byte_equality(mutation):
    expected = np.ones((3,2),dtype=np.float32); actual = expected.copy()
    if mutation == 'ulp': actual[0,0] = np.nextafter(actual[0,0], np.float32(2))
    elif mutation == 'dtype': actual = actual.astype(np.float64)
    elif mutation == 'shape': actual = actual.reshape(2,3)
    elif mutation == 'nan': actual[0,0] = np.nan
    else: expected[0,0] = 0.; actual[0,0] = -0.
    with pytest.raises(ValueError): replay.bitwise_array(actual, expected, np, 'fixture')


@pytest.mark.parametrize('actual,expected', [(np.nextafter(1.,2.),1.),(np.inf,np.inf),(-0.,0.)])
def test_source_scores_have_no_relaxed_tolerance(actual, expected):
    with pytest.raises(ValueError): replay.bitwise_score(actual, expected)


@pytest.mark.parametrize('field', ['algorithm','keys','position','has_gauss','cached_gaussian'])
def test_every_NumPy_state_field_required(field):
    expected = np.random.RandomState(1103).get_state()
    actual = list(expected); actual[1] = expected[1].copy()
    if field == 'algorithm': actual[0] = 'other'
    elif field == 'keys': actual[1][0] ^= np.uint32(1)
    elif field == 'position': actual[2] -= 1
    elif field == 'has_gauss': actual[3] = 1
    else: actual[4] = -0.
    with pytest.raises(ValueError): replay.numpy_state_matches(tuple(actual), expected, np)


@pytest.mark.parametrize('missing', ['model.meta','model.index','model.data-00000-of-00001','numpy_random_state.npz'])
def test_complete_native_checkpoint_files_required(missing):
    record = dict(artifacts=[dict(path='fixture/checkpoint/'+name) for name in [
        'model.meta','model.index','model.data-00000-of-00001','numpy_random_state.npz'] if name != missing])
    with pytest.raises(ValueError): replay.checkpoint_directory(record)


def report_fixture():
    queue, config = configuration(); route = config['routes'][0]; count = route['full_original_data_shape'][0]
    binding = dict(result_sha256='result', binding_sha256='binding')
    rows = [dict(kind=kind,repeat=repeat,source_protocol_iterations_reconstructed=2000 if kind == 'discriminative' else 5000,
        complete_test_windows=2*(count-int(count*.8)) if kind == 'discriminative' else count,
        all_native_variables_restored_bitwise=True,full_native_predictions_bitwise_verified=True,
        full_native_score_bitwise_verified=True,full_post_evaluator_NumPy_state_bitwise_verified=True,
        no_model_or_optimizer_variables_changed=True) for kind in ('discriminative','predictive') for repeat in range(10)]
    report = dict(passed=True,canonical_job=route['canonical_job'],execution_job=route['execution_job'],
        result_sha256='result',binding_sha256='binding',complete_checkpoint_restores=23,complete_trained_evaluator_replays=20,
        original_source_protocol_iterations_reconstructed=70000,numerical_optimizer_updates_executed=0,
        training_executed=False,additional_training_seed_slots=0,full_original_data_shape=route['full_original_data_shape'],
        full_generated_windows_replayed=count,complete_normalized_and_renormalized_generation_bitwise_verified=True,
        complete_source_mean_scores_bitwise_verified=True,independent_full_loader_bitwise_verified=True,
        stage_checkpoint_restores=[dict(tag=t,all_native_variables_restored_bitwise=True) for t in [
            'embedding_complete','supervised_complete','timegan_complete']],evaluators=rows)
    receipt = dict(exit_code=0,resource_guard_aborted=False,complete_full_native_replay_passed=True)
    return route,binding,report,receipt


def test_full_native_report_structure_required():
    replay.check_report(*report_fixture())


@pytest.mark.parametrize('mutation', ['checkpoint','critic','protocol','optimizer','generated','mean','loader','stage',
    'prediction','score','rng','weights','test_windows','guard_abort','worker_failed','missing_receipt'])
def test_any_partial_or_failed_native_replay_cannot_count_as_completion(mutation):
    route,binding,report,receipt = report_fixture()
    if mutation == 'checkpoint': report['complete_checkpoint_restores'] = 22
    elif mutation == 'critic': report['evaluators'].pop()
    elif mutation == 'protocol': report['evaluators'][0]['source_protocol_iterations_reconstructed'] = 1999
    elif mutation == 'optimizer': report['numerical_optimizer_updates_executed'] = 1
    elif mutation == 'generated': report['full_generated_windows_replayed'] -= 1
    elif mutation == 'mean': report['complete_source_mean_scores_bitwise_verified'] = False
    elif mutation == 'loader': report['independent_full_loader_bitwise_verified'] = False
    elif mutation == 'stage': report['stage_checkpoint_restores'][0]['all_native_variables_restored_bitwise'] = False
    elif mutation == 'prediction': report['evaluators'][0]['full_native_predictions_bitwise_verified'] = False
    elif mutation == 'score': report['evaluators'][0]['full_native_score_bitwise_verified'] = False
    elif mutation == 'rng': report['evaluators'][0]['full_post_evaluator_NumPy_state_bitwise_verified'] = False
    elif mutation == 'weights': report['evaluators'][0]['no_model_or_optimizer_variables_changed'] = False
    elif mutation == 'test_windows': report['evaluators'][0]['complete_test_windows'] -= 1
    elif mutation == 'guard_abort': receipt['resource_guard_aborted'] = True
    elif mutation == 'worker_failed': receipt['exit_code'] = 1
    else: receipt.pop('complete_full_native_replay_passed')
    with pytest.raises(ValueError): replay.check_report(route,binding,report,receipt)


def author_protocol_fixture(kind, tf, tokens):
    """Keep the entire real source sampling/metric tail, replacing only graphs.

    This synthetic fixture runs all2000/5000 draws. It does not train a model,
    use original data, restore real weights or establish benchmark equivalence.
    """
    native = ast.parse(native_source(kind).read_text(encoding='utf-8'))
    function = next(n for n in native.body if isinstance(n,ast.FunctionDef) and n.name == kind+'_score_metrics')
    cutoff = next(i for i,n in enumerate(function.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id == 'sess' for t in n.targets))
    tail = copy.deepcopy(function.body[cutoff:])
    iterations = 2000 if kind == 'discriminative' else 5000
    header = ast.parse('no,seq_len,dim=np.asarray(ori_data).shape\n'
        'ori_time,ori_max_seq_len=extract_time(ori_data)\n'
        'generated_time,generated_max_seq_len=extract_time(ori_data)\n'
        'iterations=%d\nbatch_size=128\n'%iterations + '\n'.join(name+"=tokens['"+name+"']" for name in tokens)).body
    variant = copy.deepcopy(function); variant.body = header+tail
    utilities = replay.ROOT/'data/public_datasets/flow_matching/timegan_native_source_v1/original/utils.py'
    helpers = [n for n in ast.parse(utilities.read_text(encoding='utf-8')).body
        if isinstance(n,ast.FunctionDef) and n.name in ('train_test_divide','extract_time','batch_generator')]
    namespace = dict(np=np,tf=tf,tokens=tokens,accuracy_score=accuracy_score,mean_absolute_error=mean_absolute_error)
    exec(compile(ast.fix_missing_locations(ast.Module(body=helpers+[variant],type_ignores=[])),'<full-original-source-protocol-fixture>','exec'),namespace)
    return namespace[variant.name]


@pytest.mark.parametrize('kind', ['discriminative','predictive'])
def test_production_replay_session_reconstructs_full_original_draw_protocol_and_targets(kind,tmp_path):
    tokens = {name:SimpleNamespace(op=object()) for name in ['X','X_hat','Y','T','T_hat','y_pred',
        'y_pred_real','y_pred_fake','d_solver','d_loss','p_solver','p_loss']}
    class Operation:
        type='NoOp'; name='init'
    saved = {}; source_updates = [0]

    class FixedFixtureModel:
        def __init__(self,*args,**kwargs): pass
        def run(self,fetches,*args,**kwargs):
            if isinstance(fetches,Operation): return None
            feed = kwargs['feed_dict']
            if isinstance(fetches,list) and fetches[0] in (tokens['d_solver'],tokens['p_solver']):
                source_updates[0] += 1
                return [None,0.0]
            x=np.asarray(feed[tokens['X']])
            if kind == 'discriminative':
                fake=np.asarray(feed[tokens['X_hat']])
                result=[x.mean((1,2)).astype(np.float32).reshape(-1,1),fake.mean((1,2)).astype(np.float32).reshape(-1,1)]
                saved['real_probabilities'],saved['synthetic_probabilities']=result
                return result
            result=x.mean(2,keepdims=True).astype(np.float32)
            saved['predictions']=result
            caller=inspect.currentframe().f_back.f_locals
            if 'Y_mb' in caller:
                saved['targets']=np.asarray(caller['Y_mb'])
            return result

    # Tokens are lightweight hashable objects, not TensorFlow or trained weights.
    class Token:
        def __init__(self): self.op=object()
    tokens = {name:Token() for name in tokens}
    tf=SimpleNamespace(Session=FixedFixtureModel,Operation=Operation,
        global_variables_initializer=lambda:Operation())
    fixture=author_protocol_fixture(kind,tf,tokens)
    real=np.random.RandomState(29).uniform(size=(160,24,6))
    generated=np.random.RandomState(31).uniform(size=(160,24,6))
    np.random.seed(1103)
    baseline_score=fixture(real,generated)
    final_state=np.random.get_state()
    iterations=2000 if kind=='discriminative' else 5000
    assert source_updates[0] == iterations
    native_output=tmp_path/'native';native_output.mkdir()
    np.savez_compressed(native_output/('%s_00_full_test.npz'%kind),**saved)
    nodes=ast.parse(inspect.getsource(replay.worker))
    replay_class=next(n for n in ast.walk(nodes) if isinstance(n,ast.ClassDef) and n.name=='ReplayedSession')
    disc=ModuleType('fixture_disc');disc.__file__=str(native_source('discriminative'))
    pred=ModuleType('fixture_pred');pred.__file__=str(native_source('predictive'))
    context=dict(kind=kind,repeat=0,checkpoint={},session=None)
    namespace=dict(replay.__dict__,np=np,tf=tf,disc=disc,pred=pred,original_session=FixedFixtureModel,
        context=context,session_config=None,stochastic_ancestors=lambda ops:[],native_output=native_output,
        output=tmp_path/'replay',real=real,checkpoint_directory=lambda record:tmp_path/'checkpoint',
        restore_full_checkpoint=lambda *args:dict(all_native_variables_restored_bitwise=True))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[replay_class],type_ignores=[])),'<production-native-replay-session-fixture>','exec'),namespace)
    tf.Session=namespace['ReplayedSession']
    source_updates[0]=0
    np.random.seed(1103)
    score=fixture(real,generated)
    replay.bitwise_score(score,baseline_score)
    replay.numpy_state_matches(np.random.get_state(),final_state,np)
    session=context['session']
    assert session.iterations==iterations and session.predictions==1
    assert source_updates[0]==0  # Production session suppresses numerical updates.
    assert session.evidence['full_native_predictions_bitwise_verified']
    assert session.evidence['complete_test_windows']==(64 if kind=='discriminative' else 160)
