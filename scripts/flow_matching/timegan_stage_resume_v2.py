"""Verified native TimeGAN stage recovery; Python 3.7 compatible.

Author reference: Yoon, Jarrett and van der Schaar, NeurIPS 2019.
Only the two completed pretraining loops are omitted in memory. The full
original graph, joint loop, generation and twenty evaluator trainings remain.
"""
import ast
import copy
import inspect
from pathlib import Path

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha, verify_queue


def binding(config):
    queue = read(ROOT / config['original_queue'])
    if sha(ROOT / config['original_queue']) != config['original_queue_sha256']:
        raise ValueError('Original complete 45-job queue changed')
    model = verify_queue(queue)
    job = next(j for j in queue['jobs'] if j['id'] == config['canonical_job_id'])
    if job != config['canonical_job']:
        raise ValueError('Recovery must occupy the same canonical training seed')
    previous = ROOT / job['output_directory']
    records = []
    for tag, counts in [('embedding_complete', dict(E0=50000, GS=0, G=0, E=0, D=0)),
                        ('supervised_complete', dict(E0=50000, GS=50000, G=0, E=0, D=0))]:
        path = previous / 'checkpoints' / tag / 'receipt.json'
        if sha(path) != config['stage_receipt_sha256'][tag]:
            raise ValueError('Completed stage receipt changed')
        record = read(path)
        if record['optimizer_updates'] != counts or record['tag'] != tag or record['kind'] != 'timegan':
            raise ValueError('Full completed original pretraining stages required')
        for artifact in record['artifacts']:
            if sha(artifact['path']) != artifact['sha256']:
                raise ValueError('Completed native checkpoint or RNG state changed')
        records.append(record)
    if records[0]['model_parameters'] != records[1]['model_parameters']:
        raise ValueError('Original full parameter identity differs between stages')
    if (ROOT / config['output_directory']).resolve() == previous.resolve():
        raise ValueError('Never overwrite the interrupted attempt')
    return queue, model, job, records


def resumed_ast(module):
    tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'timegan')
    variant = copy.deepcopy(function)
    loops = [i for i, n in enumerate(variant.body) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == 'itt']
    if len(loops) != 3:
        raise ValueError('Exact original three training stages required')
    original_joint = ast.dump(variant.body[loops[2]])
    for index in loops[:2]:
        variant.body[index] = ast.copy_location(ast.Pass(), variant.body[index])
    variant.body.insert(loops[0], ast.parse('_verified_restore(sess, locals())').body[0])
    remaining = [n for n in variant.body if isinstance(n, ast.For)
                 and isinstance(n.target, ast.Name) and n.target.id == 'itt']
    if len(remaining) != 1 or ast.dump(remaining[0]) != original_joint:
        raise ValueError('Original complete joint training loop changed')
    variant.name = 'timegan_verified_stage_resume'
    unit = ast.Module(body=[variant]) if __import__('sys').version_info < (3, 8) else ast.Module(body=[variant], type_ignores=[])
    return ast.fix_missing_locations(unit)


def resumed_function(module, restore):
    unit = resumed_ast(module)
    namespace = dict(module.__dict__, _verified_restore=restore)
    exec(compile(unit, module.__file__, 'exec'), namespace)
    return namespace[unit.body[0].name]


def numpy_state(path):
    import numpy as np
    with np.load(str(path), allow_pickle=False) as state:
        return (str(state['algorithm'].item()), state['keys'].copy(), int(state['position']),
                int(state['has_gauss']), float(state['cached_gaussian']))


def stochastic_ancestors(operations):
    pending = list(operations)
    seen = set()
    random = []
    while pending:
        op = pending.pop()
        if op.name in seen:
            continue
        seen.add(op.name)
        if 'Random' in op.type or op.type in ('Multinomial', 'RandomShuffle'):
            random.append(dict(name=op.name, type=op.type))
        pending.extend(t.op for t in op.inputs)
        pending.extend(op.control_inputs)
    return random


def restore_all(session, nodes, checkpoint, rng_path):
    import numpy as np
    import tensorflow as tf
    variables = tf.global_variables()
    reader = tf.train.NewCheckpointReader(str(checkpoint))
    actual = {v.op.name: v.shape.as_list() for v in variables}
    if actual != reader.get_variable_to_shape_map():
        raise ValueError('Every native variable and optimizer slot must be restored')
    random = stochastic_ancestors([nodes[k + '_solver'] for k in ('E0', 'E', 'D', 'G', 'GS')]
                                 + [nodes['X_hat'].op])
    if random:
        raise ValueError('Checkpoint does not preserve stochastic TF execution state: ' + str(random))
    tf.train.Saver(var_list=variables).restore(session, str(checkpoint))
    arrays = session.run(variables)
    if any(not np.array_equal(a, reader.get_tensor(v.op.name)) or not np.isfinite(a).all()
           for v, a in zip(variables, arrays)):
        raise ValueError('Restored all-variable checkpoint values differ')
    np.random.set_state(numpy_state(rng_path))
    return dict(restored_native_variables=len(variables), all_variable_values_bitwise_equal=True,
                all_variables_finite=True, stochastic_training_or_generation_ancestors=random,
                checkpoint_index_sha256=sha(str(checkpoint) + '.index'), rng_sha256=sha(rng_path))


def compile_observed_resume(config):
    """Adapt the existing observer in memory; never modify a running worker's file."""
    from scripts.flow_matching import run_timegan_native_job_v1 as base
    queue, model, original_job, stage_records = binding(config)
    proof_path = ROOT / config['native_restore_audit']
    if sha(proof_path) != config['native_restore_audit_sha256']:
        raise ValueError('Independent full native restore proof changed')
    proof = read(proof_path)
    if not proof['passed'] or proof['canonical_job_id'] != original_job['id']:
        raise ValueError('Recovery requires the full native restore proof')
    prior = stage_records[-1]['optimizer_updates']

    def make_resumed(native, output):
        module = __import__(native.__module__, fromlist=['timegan'])

        def restore(session, nodes):
            directory = ROOT / original_job['output_directory'] / 'checkpoints/supervised_complete'
            evidence = restore_all(session, nodes, directory / 'model', directory / 'numpy_random_state.npz')
            # Counts are a cumulative ledger, explicitly split into prior/fresh updates below.
            session.counts = dict(prior)
            evidence.update(restored_completed_stage_updates=prior, fresh_updates_at_restore=0,
                            canonical_job_id=original_job['id'], independent_seed_slots_added=0)
            write(output / 'resume_restore_verification.json', evidence)

        return resumed_function(module, restore)

    tree = ast.parse(inspect.getsource(base.run))
    function = tree.body[0]
    matches = 0
    for index, node in enumerate(function.body):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'native_timegan' for t in node.targets):
            function.body[index:index + 1] = ast.parse(
                'native_timegan = _make_resumed(author.timegan, output)\n'
                'checkpoints.extend(_stage_records)').body
            matches += 1
            break
    if matches != 1:
        raise ValueError('Exactly one observer/native function binding required')
    final_dicts = [n for n in ast.walk(function) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name) and n.func.id == 'dict'
                   and any(k.arg == 'full_original_source_main_completed' for k in n.keywords)]
    if len(final_dicts) != 1:
        raise ValueError('Exactly one final full native result required')
    result = final_dicts[0]
    next(k for k in result.keywords if k.arg == 'full_original_source_main_completed').value = ast.NameConstant(value=False)
    additions = ast.parse("dict(source_main_with_verified_stage_resume_completed=True, "
        "restored_completed_stage_updates=_prior, fresh_optimizer_updates={k: model_counts[k]-_prior[k] for k in model_counts}, "
        "resume_provenance=_provenance)", mode='eval').body.keywords
    result.keywords.extend(additions)
    namespace = dict(base.__dict__, _make_resumed=make_resumed, _stage_records=stage_records, _prior=prior,
        _provenance=dict(original_job=original_job, original_attempt_preserved=True,
            resume_config=config, prior_stage_receipts=config['stage_receipt_sha256'],
            native_restore_audit=str(proof_path), native_restore_audit_sha256=sha(proof_path),
            discarded_unsaved_joint_updates_not_counted=True, independent_training_seed_slots_added=0))
    exec(compile(ast.fix_missing_locations(tree), str(Path(base.__file__)), 'exec'), namespace)
    job = dict(original_job, output_directory=config['output_directory'])
    return namespace['run'], queue, job


def verify_recovery_result(config):
    from scripts.flow_matching.run_timegan_native_job_v1 import validate_evaluators
    from scripts.flow_matching.timegan_native_common_v1 import verify_counts
    queue, model, job, stages = binding(config)
    result = read(ROOT / config['output_directory'] / 'result.json')
    verify_counts(result['optimizer_updates'], model)
    validate_evaluators(result['evaluators'])
    prior = stages[-1]['optimizer_updates']
    if (result['job'] != dict(job, output_directory=config['output_directory'])
            or result['full_original_source_main_completed']
            or not result['source_main_with_verified_stage_resume_completed']
            or not result['full_native_three_training_stages_completed']
            or not result['full_native_evaluator_trainings_completed']
            or result['restored_completed_stage_updates'] != prior
            or result['fresh_optimizer_updates'] != {k: result['optimizer_updates'][k] - prior[k] for k in prior}
            or len(result['checkpoints']) != 23 or len(result['artifacts']) != 22
            or len(result['visualizations']) != 2 or len(result['graph_seeds']) != 21
            or result['queue_sha256'] != config['original_queue_sha256']
            or result['independent_model_training_seed_slots'] != 1
            or result['resume_provenance']['independent_training_seed_slots_added'] != 0
            or result['author_equivalence_certified'] or result['strict_TAB_result']):
        raise ValueError('Complete full-budget resumed native experiment required')
    shape = read(ROOT / model['dataset_config'])['datasets'][job['dataset']]['full_shape']
    graph_audit = read(ROOT / queue['construction_proof'])
    expected_parameters = next(r['model_parameters'] for r in graph_audit['records']
                              if r['dataset'] == job['dataset'] and r['module'] == job['module'])
    if result['full_original_data_shape'] != shape or result['full_generated_shape'] != shape:
        raise ValueError('Original full native dataset/generation shape required')
    if (result['model_parameters'] != expected_parameters
            or result['environment_lock_sha256'] != sha(ROOT / queue['environment_lock'])
            or result['resume_provenance']['resume_config'] != config
            or [r['seed'] for r in result['graph_seeds']] != [job['seed'] + 1009 * i for i in range(21)]):
        raise ValueError('Frozen full graph/environment/resume/seed identity required')
    for artifact in result['artifacts'] + result['visualizations'] + [a for c in result['checkpoints'] for a in c['artifacts']]:
        if sha(artifact['path']) != artifact['sha256']:
            raise ValueError('Resumed native artifact changed')
    if result['checkpoints'][:2] != stages:
        raise ValueError('Prior checkpoints must be attributed without fabrication')
    return result
