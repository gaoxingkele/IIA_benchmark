"""Complete native predictive evaluations after a failed observation attempt.

TimeGAN, Yoon et al., NeurIPS 2019, https://github.com/jsyoon0823/TimeGAN.
Successful full generator and discriminative training is attributed to the
preserved predecessor. Only unsaved predictors are trained again. No author
model, loss, optimizer, data loader or predictive evaluator is edited.
Python 3.7 compatible; independent full checkpoint replay remains mandatory.
"""
import ast
import copy
import inspect
import math
from pathlib import Path
import shutil

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha, verify_queue, verify_counts
from scripts.flow_matching.timegan_predictive_observer_v2 import corrected_observer_ast, native_predictive_targets
from scripts.flow_matching.timegan_stage_resume_v2 import numpy_state


def graph_record(seed, index):
    return dict(index=index, kind='timegan' if index == 0 else 'discriminative',
                repeat=None if index == 0 else index - 1, seed=seed + 1009 * index)


def binding(config):
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen posthoc recovery source changed: ' + pin['path'])
    if sha(ROOT / config['original_queue']) != config['original_queue_sha256']:
        raise ValueError('Original complete 45-job queue changed')
    queue = read(ROOT / config['original_queue'])
    model = verify_queue(queue)
    job = next(j for j in queue['jobs'] if j['id'] == config['canonical_job']['id'])
    if (job != config['canonical_job'] or config['settings'] != queue['settings']
            or config['resource_lock'] != queue['resource_lock'] or config['python'] != queue['python']
            or config['cpu_threads'] != queue['cpu_threads']
            or config['original_scope'] != dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273)):
        raise ValueError('Same canonical seed and unchanged full scope/resource guards required')
    previous = ROOT / job['output_directory']
    destination = (ROOT / config['output_directory']).resolve()
    if destination == previous.resolve() or previous.resolve() in destination.parents:
        raise ValueError('Keep predecessor attempt entirely untouched')
    if (previous / 'result.json').exists():
        raise ValueError('Do not recover a job that already has a complete result')
    terminal = ROOT / config['terminal_receipt']['path']
    if sha(terminal) != config['terminal_receipt']['sha256']:
        raise ValueError('Original terminal receipt changed')
    receipt = read(terminal)
    if (receipt.get('exit_code') != 1 or receipt.get('resource_guard_aborted') is not False
            or receipt.get('full_native_result_verified') is not False):
        raise ValueError('Exact preserved observer failure required')
    failure = ROOT / config['failure_log']['path']
    if sha(failure) != config['failure_log']['sha256'] or "KeyError: <tf.Tensor 'myinput_y:0'" not in failure.read_text(encoding='utf-8'):
        raise ValueError('Exact missing predictor Y feed failure required')
    tags = ['embedding_complete', 'supervised_complete', 'timegan_complete'] + [
        'discriminative_%02d_complete' % i for i in range(10)]
    if [p['tag'] for p in config['checkpoint_receipts']] != tags:
        raise ValueError('All three successful generator stages and ten discriminators required')
    records = []
    for pin in config['checkpoint_receipts']:
        path = previous / 'checkpoints' / pin['tag'] / 'receipt.json'
        if sha(path) != pin['sha256']:
            raise ValueError('Successful predecessor checkpoint receipt changed')
        record = read(path)
        if record['tag'] != pin['tag']:
            raise ValueError('Exact predecessor checkpoint tag required')
        directory = path.parent.resolve()
        names = {Path(a['path']).name for a in record['artifacts']}
        if names != {'checkpoint', 'model.data-00000-of-00001', 'model.index', 'model.meta', 'numpy_random_state.npz'}:
            raise ValueError('Complete native model, optimizer and NumPy checkpoint required')
        for artifact in record['artifacts']:
            if Path(artifact['path']).resolve().parent != directory or sha(artifact['path']) != artifact['sha256']:
                raise ValueError('Preserved native checkpoint artifact changed')
        records.append(record)
    verify_counts(records[2]['optimizer_updates'], model)
    expected = [dict(E0=50000, GS=0, E=0, G=0, D=0), dict(E0=50000, GS=50000, E=0, G=0, D=0), records[2]['optimizer_updates']]
    for index, record in enumerate(records):
        if index < 3:
            if (record['kind'] != 'timegan' or record['optimizer_updates'] != expected[index]
                    or record['model_parameters'] != records[2]['model_parameters']):
                raise ValueError('Exact full original generator stages required')
        elif record['kind'] != 'discriminative' or record['optimizer_updates'] != {'evaluator': 2000}:
            raise ValueError('Full original 2000-update discriminator required')
    evaluator_path = previous / 'completed_evaluators.json'
    if sha(evaluator_path) != config['completed_evaluators_sha256']:
        raise ValueError('Preserved complete discriminative records changed')
    evaluators = read(evaluator_path)
    if len(evaluators) != 10:
        raise ValueError('Exactly all ten saved discriminators and no saved predictor required')
    for index, record in enumerate(evaluators):
        if (record['kind'] != 'discriminative' or record['repeat'] != index
                or record['optimizer_updates'] != 2000 or record['graph_seed'] != job['seed'] + 1009 * (index + 1)
                or not math.isfinite(record['score'])):
            raise ValueError('Every original discriminator repeat, budget and seed required')
    expected_archives = ['full_native_generation_feed.npz', 'full_native_real_and_generated.npz'] + [
        'discriminative_%02d_full_test.npz' % i for i in range(10)]
    if [p['name'] for p in config['archive_receipts']] != expected_archives:
        raise ValueError('Both full generation arrays and all ten test probability archives required')
    for pin in config['archive_receipts']:
        if sha(previous / pin['name']) != pin['sha256']:
            raise ValueError('Preserved full prediction/generation archive changed')
    return queue, model, job, records, evaluators


def verify_saved_arrays(config, model, job, evaluators, np):
    """Full-array receipt/score audit; explicitly not checkpoint re-inference."""
    previous = ROOT / job['output_directory']
    shape = read(ROOT / model['dataset_config'])['datasets'][job['dataset']]['full_shape']
    from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import bitwise_array, bitwise_score
    with np.load(str(previous / 'full_native_real_and_generated.npz'), allow_pickle=False) as archive:
        if set(archive.files) != {'real', 'generated'}:
            raise ValueError('Exact complete real/generated archive required')
        real, generated = archive['real'].copy(), archive['generated'].copy()
    if list(real.shape) != shape or generated.shape != real.shape or not np.isfinite(real).all() or not np.isfinite(generated).all():
        raise ValueError('Every full original finite real/generated window required')
    with np.load(str(previous / 'full_native_generation_feed.npz'), allow_pickle=False) as feed:
        if set(feed.files) != {'noise', 'normalized_real', 'times', 'normalized_generated', 'min_val', 'max_val'}:
            raise ValueError('Exact complete generation feed required')
        if any(not np.isfinite(feed[k]).all() for k in feed.files):
            raise ValueError('Every preserved generation feed coordinate must be finite')
        if any(list(feed[k].shape) != shape for k in ('noise', 'normalized_real', 'normalized_generated')):
            raise ValueError('All complete native generation shapes required')
        if feed['times'].shape != (shape[0],) or not np.all(feed['times'] == shape[1]):
            raise ValueError('Every complete original window length required')
        minimum, scale = np.min(real, axis=(0, 1)), np.max(real, axis=(0, 1)) - np.min(real, axis=(0, 1))
        bitwise_array(minimum, feed['min_val'], np, 'preserved full data minimum')
        bitwise_array(scale, feed['max_val'], np, 'preserved full data scale')
        bitwise_array((real - minimum) / (scale + 1e-7), feed['normalized_real'], np, 'preserved full normalization')
        bitwise_array(feed['normalized_generated'] * scale + minimum, generated, np, 'preserved full renormalization')
    from sklearn.metrics import accuracy_score
    count = shape[0] - int(shape[0] * .8)
    for index, record in enumerate(evaluators):
        with np.load(str(previous / ('discriminative_%02d_full_test.npz' % index)), allow_pickle=False) as archive:
            if set(archive.files) != {'real_probabilities', 'synthetic_probabilities'}:
                raise ValueError('Exact complete discriminator probabilities required')
            values = [archive[k] for k in ('real_probabilities', 'synthetic_probabilities')]
            if any(v.shape != (count, 1) or not np.isfinite(v).all() or (v < 0).any() or (v > 1).any() for v in values):
                raise ValueError('Both complete original discriminator test groups required')
            accuracy = accuracy_score(np.concatenate([np.ones((count,)), np.zeros((count,))]),
                                      np.squeeze(np.concatenate(values)) > .5)
            bitwise_score(abs(.5 - accuracy), record['score'])
    return real, generated


def posthoc_ast():
    from scripts.flow_matching import run_timegan_native_job_v1 as base
    tree = corrected_observer_ast(ast.parse(inspect.getsource(base.run)))
    function = tree.body[0]
    replacements = 0
    for node in function.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'checkpoints' for t in node.targets):
            node.value = ast.parse('copy.deepcopy(_prior_checkpoints)', mode='eval').body
            replacements += 1
        if isinstance(node, ast.FunctionDef) and node.name == 'complete_timegan':
            node.body = ast.parse(
                'nonlocal model_counts, model_parameters\n'
                "context.update(kind='timegan', repeat=None, session=None)\n"
                'generated = _reuse_generator(data, parameters, output, context)\n'
                "model_counts = dict(_prior_checkpoints[2]['optimizer_updates'])\n"
                "model_parameters = _prior_checkpoints[2]['model_parameters']\n"
                'return generated').body
            replacements += 1
    evaluator = next(n for n in ast.walk(function) if isinstance(n, ast.FunctionDef) and n.name == 'evaluator')
    # The entire original predictive training/observation/checkpoint path is retained.
    evaluator.body.insert(0, ast.parse("if kind == 'discriminative':\n return _reuse_discriminator(data, generated, output, context)").body[0])
    finals = [n for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
              and n.func.id == 'dict' and any(k.arg == 'full_original_source_main_completed' for k in n.keywords)]
    if replacements != 2 or len(finals) != 1:
        raise ValueError('Exact preserved original observer structure required')
    next(k for k in finals[0].keywords if k.arg == 'full_original_source_main_completed').value = ast.NameConstant(value=False)
    finals[0].keywords.extend(ast.parse(
        "dict(source_main_with_verified_posthoc_recovery_completed=True, posthoc_recovery_provenance=_provenance, "
        "reused_generator_optimizer_updates=model_counts, fresh_generator_optimizer_updates={k: 0 for k in model_counts}, "
        "reused_discriminative_optimizer_updates=20000, fresh_discriminative_optimizer_updates=0, "
        "fresh_predictive_optimizer_updates=50000, discarded_unsaved_predictor_updates_not_credited=True)", mode='eval').body.keywords)
    return ast.fix_missing_locations(tree)


def compile_recovery(config):
    from scripts.flow_matching import run_timegan_native_job_v1 as base
    from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import bitwise_array
    queue, model, job, checkpoints, evaluators = binding(config)
    cached = {}

    def reuse_generator(data, parameters, output, context):
        import numpy as np
        if context['graph_seeds'] or context['completed_evaluators'] or parameters != dict(
                module=job['module'], hidden_dim=24, num_layer=3, iterations=50000, batch_size=128):
            raise ValueError('Only the exact original initial generator call may reuse preserved training')
        real, generated = verify_saved_arrays(config, model, job, evaluators, np)
        bitwise_array(data, real, np, 'current untouched native loader versus all saved real windows')
        for pin in config['archive_receipts']:
            destination = output / pin['name']
            if destination.exists():
                raise FileExistsError('Never overwrite recovery arrays')
            shutil.copyfile(str(ROOT / job['output_directory'] / pin['name']), str(destination))
            if sha(destination) != pin['sha256']:
                raise ValueError('Copied complete native artifact differs')
        cached.update(real=real, generated=generated)
        context['graph_seeds'].append(graph_record(job['seed'], 0))
        return generated

    def reuse_discriminator(data, generated, output, context):
        import numpy as np
        repeat = len(context['completed_evaluators'])
        if repeat >= 10 or len(context['graph_seeds']) != repeat + 1:
            raise ValueError('Every completed discriminator must be consumed once, in original order')
        bitwise_array(data, cached['real'], np, 'every reused discriminator real input')
        bitwise_array(generated, cached['generated'], np, 'every reused discriminator generated input')
        record = copy.deepcopy(evaluators[repeat])
        context.update(kind='discriminative', repeat=repeat, session=None)
        context['graph_seeds'].append(graph_record(job['seed'], repeat + 1))
        context['completed_evaluators'].append(record)
        write(output / 'completed_evaluators.json', context['completed_evaluators'])
        # Restore the original post-evaluation NumPy state, not a fresh seed.
        np.random.set_state(numpy_state(ROOT / job['output_directory'] / 'checkpoints' /
                                      ('discriminative_%02d_complete' % repeat) / 'numpy_random_state.npz'))
        return record['score']

    namespace = dict(base.__dict__, copy=copy, _native_predictive_targets=native_predictive_targets,
        _prior_checkpoints=checkpoints, _reuse_generator=reuse_generator, _reuse_discriminator=reuse_discriminator,
        _provenance=dict(configuration=config, canonical_job=job, predecessor_preserved=True,
            reused_graph_indices=list(range(11)), newly_executed_graph_indices=list(range(11, 21)),
            predecessor_terminal_receipt=config['terminal_receipt'], independent_training_seed_slots_added=0,
            independent_full_checkpoint_replay_required=True, source_training_or_scoring_code_changed=False))
    exec(compile(posthoc_ast(), base.__file__, 'exec'), namespace)
    return namespace['run'], queue, dict(job, output_directory=config['output_directory'])


def verify_recovery_result(config):
    from scripts.flow_matching.run_timegan_native_job_v1 import validate_evaluators
    from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import complete_metadata, bitwise_score
    queue, model, job, checkpoints, evaluators = binding(config)
    result = read(ROOT / config['output_directory'] / 'result.json')
    execution = dict(job, output_directory=config['output_directory'])
    route = dict(kind='posthoc_recovery', canonical_job=job, execution_job=execution)
    complete_metadata(route, queue, result, dict(exit_code=0, resource_guard_aborted=False, verified=True), 'verified')
    if (result.get('full_original_source_main_completed') is not False
            or result.get('source_main_with_verified_posthoc_recovery_completed') is not True
            or result['checkpoints'][:13] != checkpoints or result['evaluators'][:10] != evaluators
            or result.get('queue_sha256') != config['original_queue_sha256']
            or result.get('environment_lock_sha256') != sha(ROOT / queue['environment_lock'])
            or result.get('reused_generator_optimizer_updates') != checkpoints[2]['optimizer_updates']
            or result.get('fresh_generator_optimizer_updates') != {k: 0 for k in checkpoints[2]['optimizer_updates']}
            or result.get('reused_discriminative_optimizer_updates') != 20000
            or result.get('fresh_discriminative_optimizer_updates') != 0
            or result.get('fresh_predictive_optimizer_updates') != 50000
            or result.get('discarded_unsaved_predictor_updates_not_credited') is not True
            or result['posthoc_recovery_provenance']['configuration'] != config
            or result['posthoc_recovery_provenance']['independent_training_seed_slots_added'] != 0
            or result.get('independent_full_checkpoint_replay_completed') is not False):
        raise ValueError('Exact preserved predecessor plus all full fresh predictor trainings required')
    validate_evaluators(result['evaluators'])
    for artifact in result['artifacts'] + result['visualizations'] + [a for c in result['checkpoints'] for a in c['artifacts']]:
        if sha(artifact['path']) != artifact['sha256']:
            raise ValueError('Complete recovered native artifact changed')
    import numpy as np
    verify_saved_arrays(config, model, job, evaluators, np)
    for kind in ('discriminative', 'predictive'):
        bitwise_score(float(np.mean([r['score'] for r in result['evaluators'] if r['kind'] == kind])), result['scores'][kind])
    return result
