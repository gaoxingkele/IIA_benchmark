"""Preserve full original stock commands while recording all sampler/critic state.

Author references: Jeon et al., Generative Time-series GAN, NeurIPS2022.
Only observational calls and result fields are added to the frozen runner in
memory. Native modules, arguments, optimizer loops and returned scores remain.
"""
import ast
import functools
import inspect
import json
from pathlib import Path

from scripts.flow_matching import run_gtgan_native_job_v1 as original
from scripts.flow_matching.gtgan_energy_metric_receipts_v1 import observe_evaluator

ROOT, sha, full_budget = original.ROOT, original.sha, original.full_budget


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def extended_budget(case):
    expected = full_budget(case)
    if case['parsed_original_arguments']['input_size'] != 24:
        raise ValueError('Complete original latent generator input width24 required')
    return dict(expected, discriminative_evaluator_steps=2000, predictive_evaluator_steps=5000,
        latent_noise_variables=case['parsed_original_arguments']['input_size'],
        predictive_output_variables=1 if case['missing_rate'] == 0 else 6,
        predictive_function='predictive_score_metrics2' if case['missing_rate'] == 0 else 'predictive_score_metrics')


def attach_observers(module, runtime, expected, queue, np, torch, sampler_inputs, metric_receipts):
    native_sampler = module.run_model

    @functools.wraps(native_sampler)
    def observed_sampler(*positional, **keywords):
        data = positional[2]
        if keywords.get('z') is True and data.shape[0] == expected['full_generation_windows']:
            caller = inspect.currentframe().f_back.f_locals
            observations = caller.get('obs')
            if observations is None or tuple(observations.shape) != (expected['full_generation_windows'], 24):
                raise ValueError('All original recovery observation times required')
            if (tuple(data.shape) != (expected['full_generation_windows'],24,expected['latent_noise_variables'])
                    or tuple(positional[3].shape) != (expected['full_generation_windows'],24,1)):
                raise ValueError('Exact original full latent noise and conditioning time dimensions required')
            path = runtime / ('full_sampler_inputs_' + str(len(sampler_inputs) + 1) + '.npz')
            states = {'cuda_rng_' + str(i): state.cpu().numpy() for i, state in enumerate(torch.cuda.get_rng_state_all())}
            with path.open('xb') as stream:
                np.savez_compressed(stream, latent_noise=data.detach().cpu().numpy(),
                    conditioning_times=positional[3].detach().cpu().numpy(),
                    recovery_observation_times=observations.detach().cpu().numpy(),
                    torch_cpu_rng=torch.get_rng_state().numpy(), **states)
            sampler_inputs.append(dict(path=str(path), sha256=sha(path),
                native_generator_training_mode=bool(positional[1].training),
                native_recovery_training_mode=bool(caller['recovery'].training),
                native_grad_enabled=bool(torch.is_grad_enabled()), full_windows=expected['full_generation_windows']))
        return native_sampler(*positional, **keywords)

    module.run_model = observed_sampler
    for name, role in [('discriminative_score_metrics', 'discriminative'), (expected['predictive_function'], 'predictive')]:
        native = getattr(module, name)

        def bind(function, function_name, metric_role):
            @functools.wraps(function)
            def evaluate(*positional, **keywords):
                folder = runtime / 'native_evaluator_receipts' / ('evaluator_' + str(len(metric_receipts) + 1))

                def call():
                    return function(*positional, **keywords)

                call.native_function_name = function_name
                score, receipt = observe_evaluator(call, metric_role, folder,
                    expected[metric_role + '_evaluator_steps'], queue['cpu_threads'])
                metric_receipts.append(receipt)
                return score
            return evaluate

        setattr(module, name, bind(native, name, role))


def periodic_observation(index, samplers, evaluators, expected):
    if len(samplers) != index + 1 or len(evaluators) != 20 * (index + 1):
        raise ValueError('Every full native generation and all20 complete trained evaluator receipts required')
    selected = evaluators[-20:]
    roles = ['discriminative'] * 10 + ['predictive'] * 10
    if [row['role'] for row in selected] != roles or samplers[-1]['full_windows'] != expected['full_generation_windows']:
        raise ValueError('Exact original full generation and evaluator order required')
    for row in selected:
        if row['optimizer_updates'] != expected[row['role'] + '_evaluator_steps']:
            raise ValueError('Complete original 2000/5000 evaluator training required')
    return dict(sampler_input=samplers[-1], evaluator_training_receipts=selected)


def verify_periodic_artifacts(row, expected):
    """Verify full coverage and source scores; saved states still need independent replay."""
    import numpy as np
    from scripts.flow_matching.gtgan_energy_metric_receipts_v1 import recompute_score
    n, length, variables = expected['full_generation_windows'], expected['sequence_length'], expected['variables']
    sampler = row['sampler_input']
    if sha(sampler['path']) != sampler['sha256']:
        raise ValueError('Complete native noise/time/RNG snapshot changed')
    with np.load(sampler['path'],allow_pickle=False) as saved:
        shapes = {'latent_noise':(n,length,expected['latent_noise_variables']),
            'conditioning_times':(n,length,1),'recovery_observation_times':(n,length)}
        if any(saved[k].shape != v or not np.isfinite(saved[k]).all() for k,v in shapes.items()):
            raise ValueError('Every original full sampler noise/time element required')
        rng_keys = [key for key in saved.files if key.startswith('cuda_rng_')]
        if (not rng_keys or saved['torch_cpu_rng'].dtype != np.uint8
                or saved['torch_cpu_rng'].ndim != 1 or not saved['torch_cpu_rng'].size
                or any(saved[key].dtype != np.uint8 or saved[key].ndim != 1 or not saved[key].size for key in rng_keys)):
            raise ValueError('Full native CPU/CUDA RNG byte states required')
    if any(type(sampler.get(key)) is not bool for key in
            ['native_generator_training_mode','native_recovery_training_mode','native_grad_enabled']):
        raise ValueError('Exact native generator/recovery modes and gradient state required')
    with np.load(row['artifact_path'],allow_pickle=False) as generation:
        if any(generation[key].shape != (n,length,variables) or not np.isfinite(generation[key]).all()
                for key in ['real','generated']):
            raise ValueError('Every full original real/generated window required')
    scores = {'discriminative':[],'predictive':[]}
    for evaluator in row['evaluator_training_receipts']:
        role = evaluator['role']
        expected_function = 'discriminative_score_metrics' if role == 'discriminative' else expected['predictive_function']
        files = evaluator['checkpoint_files']
        names = {Path(item['path']).name for item in files}
        if (evaluator['expected_optimizer_updates'] != expected[role+'_evaluator_steps']
                or evaluator['optimizer_updates'] != expected[role+'_evaluator_steps']
                or evaluator['evaluator_function'] != expected_function
                or evaluator['full_saved_prediction_score_check_passed'] is not True
                or evaluator['additional_generator_training_seeds'] != 0
                or evaluator['evaluator_training_instances'] != 1 or evaluator['trained_variable_count'] < 1
                or names != {'native_evaluator.meta','native_evaluator.index','native_evaluator.data-00000-of-00001'}
                or len(files) != 3 or sha(evaluator['array_path']) != evaluator['array_sha256']):
            raise ValueError('All original full trained evaluator checkpoint/feed/prediction arrays required')
        for artifact in files:
            if sha(artifact['path']) != artifact['sha256']:
                raise ValueError('Complete native trained evaluator checkpoint changed')
        with np.load(evaluator['array_path'],allow_pickle=False) as arrays:
            if role == 'discriminative':
                test_n = n - int(.8*n)
                shapes = {'prediction_real':(test_n,1),'prediction_fake':(test_n,1),
                    'myinput_x:0':(test_n,length,variables),'myinput_x_hat:0':(test_n,length,variables),
                    'myinput_t:0':(test_n,),'myinput_t_hat:0':(test_n,)}
                rebuilt = recompute_score(role,[arrays['prediction_real'],arrays['prediction_fake']])
            else:
                out = expected['predictive_output_variables']
                shapes = {'prediction':(n,length-1,out),'original_reference':(n,length-1,out),
                    'myinput_x:0':(n,length-1,variables if out == variables else variables-1),'myinput_t:0':(n,)}
                rebuilt = recompute_score(role,arrays['prediction'],arrays['original_reference'])
            if any(arrays[key].shape != shape or not np.isfinite(arrays[key]).all() for key,shape in shapes.items()):
                raise ValueError('All original evaluator test feeds/predictions/targets required')
        original_score = evaluator['original_returned_score']
        if (not np.isfinite(original_score) or not np.isclose(rebuilt,original_score,rtol=1e-7,atol=1e-7)
                or rebuilt != evaluator['full_saved_predictions_recomputed_score']):
            raise ValueError('Original score and full saved predictions differ')
        scores[role].append(original_score)
    for role, values in scores.items():
        if (values != row[role+'_repeats'] or float(np.mean(values)) != row['original_metrics'][role]
                or float(np.std(values)) != row['original_metrics'][role+'_std']):
            raise ValueError('Original repeated evaluator scores/means/standard deviations differ')


def observed_main_ast(source):
    """Add only observational bindings; preserve every existing runner statement."""
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main']
    if len(tree.body) != 1:
        raise ValueError('Exactly one complete frozen native runner main required')
    function = tree.body[0]
    definitions = [i for i,n in enumerate(function.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id == 'evaluations' for t in n.targets)]
    if len(definitions) != 1:
        raise ValueError('One native evaluation ledger required')
    function.body[definitions[0]+1:definitions[0]+1] = ast.parse('sampler_inputs=[]\nmetric_receipts=[]\n_observation_budget=_extended_budget(case)').body
    bindings = [i for i,n in enumerate(function.body) if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Subscript) and ast.dump(n.targets[0]) == ast.dump(ast.parse("module.__dict__['print']=observed_print").body[0].targets[0])]
    if len(bindings) != 1:
        raise ValueError('One original print observer binding required')
    function.body.insert(bindings[0]+1, ast.parse('_attach_observers(module,runtime,_observation_budget,queue,np,torch,sampler_inputs,metric_receipts)').body[0])
    periodic = [n for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
        and n.func.id=='dict' and any(k.arg=='native_joint_step' for k in n.keywords)]
    results = [n for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
        and n.func.id=='dict' and any(k.arg=='full_native_command_completed' for k in n.keywords)]
    if len(periodic) != 1 or len(results) != 1:
        raise ValueError('One native periodic record and one complete result required')
    periodic[0].keywords.append(ast.keyword(arg=None,value=ast.parse(
        '_periodic_observation(len(evaluations),sampler_inputs,metric_receipts,_observation_budget)',mode='eval').body))
    results[0].keywords.extend(ast.parse('dict(full_native_sampler_inputs_observed=True, '
        'full_native_evaluator_trainings_observed=True, native_evaluator_receipts=metric_receipts, '
        'sampler_inputs=sampler_inputs, observer_revision="stock_full_observation_v2", '
        'observation_runtime_configuration_sha256=_observation_configuration_sha256)',mode='eval').body.keywords)
    return ast.fix_missing_locations(tree)


def compile_worker(configuration_sha256):
    namespace = dict(original.__dict__, _attach_observers=attach_observers, _extended_budget=extended_budget,
        _periodic_observation=periodic_observation, _observation_configuration_sha256=configuration_sha256)
    exec(compile(observed_main_ast(Path(original.__file__).read_text(encoding='utf-8')),original.__file__,'exec'),namespace)
    return namespace['main']


def compile_native_verifier():
    """Keep the entire original verifier without importing host-only scheduling packages."""
    path=ROOT/'scripts/flow_matching/run_gtgan_native_queue_v1.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify_result']
    if len(tree.body)!=1:
        raise ValueError('One full unchanged original stock result verifier required')
    namespace=dict(ROOT=ROOT,sha=sha,full_budget=full_budget,json=json,Path=Path)
    exec(compile(tree,str(path),'exec'),namespace)
    return namespace['verify_result']


def verify_author_environment(queue):
    lock=read(ROOT/queue['environment_lock'])
    if (original.sys.version != lock['python']
            or any(original.importlib.metadata.version(k)!=v for k,v in lock['packages'].items())):
        raise ValueError('Frozen original-version GT-GAN environment differs')


def verify_extended_result(queue, job, configuration_sha256):
    result = compile_native_verifier()(queue, job)
    model = read(ROOT / queue['model_config'])
    case = next(c for c in model['cases'] if c['id'] == job['case_id'])
    budget = extended_budget(case)
    if (result['case'] != case or result.get('observer_revision') != 'stock_full_observation_v2'
            or result['strict_TAB_result'] is not False or result['training_seeds'] != 1
            or result['native_released_pretrain_semantics_preserved'] is not True
            or not 0 <= result['optimizer_updates']['discriminator'] <= budget['maximum_discriminator_updates']
            or result.get('observation_runtime_configuration_sha256') != configuration_sha256
            or result.get('full_native_sampler_inputs_observed') is not True
            or result.get('full_native_evaluator_trainings_observed') is not True
            or len(result.get('native_evaluator_receipts',[])) != 20 * budget['full_periodic_evaluations']
            or len(result.get('sampler_inputs',[])) != budget['full_periodic_evaluations']):
        raise ValueError('All full native sampler inputs and trained evaluator states required')
    for index,row in enumerate(result['evaluations']):
        observation = periodic_observation(index, result['sampler_inputs'][:index+1],
            result['native_evaluator_receipts'][:20*(index+1)], budget)
        if row['sampler_input'] != observation['sampler_input'] or row['evaluator_training_receipts'] != observation['evaluator_training_receipts']:
            raise ValueError('Every periodic native result must bind to its actual saved sampler/critic state')
        verify_periodic_artifacts(row,budget)
    return result
