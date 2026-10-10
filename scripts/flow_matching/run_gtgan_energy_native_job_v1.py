"""Execute exact original GT-GAN Energy training and full native evaluators."""
import argparse
import builtins
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import functools
import os

from scripts.flow_matching.gtgan_energy_native_common_v1 import full_budget, verify_recipe
from scripts.flow_matching.gtgan_energy_metric_receipts_v1 import observe_evaluator

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    queue = json.loads((ROOT / args.queue).read_text(encoding='utf-8'))
    for item in queue['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen GT-GAN Energy source changed: ' + item['path'])
    lock = json.loads((ROOT / queue['environment_lock']).read_text(encoding='utf-8'))
    if sys.version != lock['python'] or any(importlib.metadata.version(k) != v for k, v in lock['packages'].items()):
        raise ValueError('Frozen original-version GT-GAN environment differs')
    construction = json.loads((ROOT / queue['construction_proof']).read_text(encoding='utf-8'))
    if (construction['full_original_constructor_cases'] != 2 or construction['performance_result']
            or not construction['passed'] or not construction['full_raw_data_processing_verified']
            or construction['config_sha256'] != queue['construction_audit_config_sha256']):
        raise ValueError('Both full original Energy data/constructor checks required')
    job = next(j for j in queue['jobs'] if j['id'] == args.job_id)
    model = json.loads((ROOT / queue['model_config']).read_text(encoding='utf-8'))
    case = next(c for c in model['cases'] + model['release_evaluation_cases'] if c['id'] == job['case_id'])
    expected = verify_recipe(case, ROOT / model['source_root'])
    runtime = Path(job['runtime_root'])
    proof = json.loads((runtime / 'runtime_preparation_receipt.json').read_text(encoding='utf-8'))
    repaired = {x['path']: x['repaired_sha256'] for x in proof['runtime_repairs']}
    if len(proof['complete_copied_assets']) != 245:
        raise ValueError('Full author runtime required')
    for item in proof['complete_copied_assets']:
        if sha(runtime / item['path']) != repaired.get(item['path'], item['original_sha256']):
            raise ValueError('Prepared native runtime changed: ' + item['path'])
    output = ROOT / job['output_directory']
    if output.exists():
        raise FileExistsError('Preserve every partial native training attempt')
    output.mkdir(parents=True)
    import numpy as np
    import torch
    import tensorflow as tf
    if not torch.cuda.is_available():
        raise RuntimeError('The unchanged native GT-GAN command requires CUDA')
    tf.config.set_visible_devices([], 'GPU')
    torch.set_num_threads(queue['cpu_threads'])
    sys.path.insert(0, str(runtime))
    file = runtime / Path(case['source_entry']).name
    spec = importlib.util.spec_from_file_location('native_gtgan_energy_training', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    phase = ['native_initialization']
    counts = dict(embedding=0, generator=0, discriminator=0)
    created = []
    evaluations = []
    metric_receipts = []
    sampler_inputs = []
    raw_print = builtins.print
    raw_adam = torch.optim.Adam

    def write_progress():
        (output / 'progress.json').write_text(json.dumps(dict(phase=phase[0], optimizer_updates=counts,
            full_periodic_evaluations_completed=len(evaluations), expected_budget=expected), indent=2) + '\n', encoding='utf-8')

    class ObservedAdam(raw_adam):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.native_role = ('embedding', 'generator', 'discriminator')[len(created)]
            created.append(self)

        def step(self, *a, **kw):
            result = super().step(*a, **kw)
            counts[self.native_role] += 1
            if sum(counts.values()) % 100 == 0:
                write_progress()
            return result

    def observed_print(*values, **kwargs):
        if values and isinstance(values[0], str):
            if values[0] == 'Start Embedding Network Training':
                phase[0] = 'native_embedding_training'
            elif values[0] == 'Start Joint Training':
                phase[0] = 'native_joint_training'
        if len(values) == 1 and isinstance(values[0], dict) and {'predictive', 'discriminative'} <= set(values[0]):
            local = inspect.currentframe().f_back.f_locals
            step = 0 if expected['native_release_checkpoint_evaluation'] else local.get('step')
            cadence = 0 if expected['native_release_checkpoint_evaluation'] else 500 * (len(evaluations) + 1)
            if step != cadence:
                raise ValueError('Complete original periodic evaluator cadence differs')
            real = np.asarray(local['generated_data1'])
            generated = np.asarray(local['generated_data2'])
            if real.shape != (19712, 24, 28) or generated.shape != real.shape or not np.isfinite(generated).all():
                raise ValueError('Full native generation window/variable coverage required')
            if len(local['predictive_score']) != 10 or len(local['discriminative_score']) != 10:
                raise ValueError('Ten full original evaluator repetitions required')
            artifact = runtime / ('full_generation_step_' + str(step) + '.npz')
            with artifact.open('xb') as stream:
                np.savez_compressed(stream, real=real, generated=generated,
                    conditioning_input=local['batch']['data'].cpu().numpy())
            current_metrics = metric_receipts[-20:]
            if (len(metric_receipts) != 20 * (len(evaluations) + 1) or len(sampler_inputs) != len(evaluations) + 1
                    or len(current_metrics) != 20):
                raise ValueError('All original evaluator training and sampler input receipts required')
            row = dict(native_joint_step=step, original_metrics={k: float(v) for k, v in values[0].items()},
                       predictive_repeats=[float(v) for v in local['predictive_score']],
                       discriminative_repeats=[float(v) for v in local['discriminative_score']],
                       full_generation_windows=19712, artifact_path=str(artifact), artifact_sha256=sha(artifact),
                       sampler_input=sampler_inputs[-1], evaluator_training_receipts=current_metrics,
                       uncertainty_scope='Evaluator repeats of one trained model, not independent training seeds')
            evaluations.append(row)
            (output / ('evaluation_' + str(step) + '.json')).write_text(json.dumps(row, indent=2) + '\n', encoding='utf-8')
            write_progress()
        raw_print(*values, **kwargs)

    sys.argv = [str(file)] + case['author_arguments'] + ['--save_dir', 'native_run_checkpoints']
    torch.optim.Adam = ObservedAdam
    module.optim.Adam = ObservedAdam
    module.__dict__['print'] = observed_print
    native_sampler = module.run_model

    @functools.wraps(native_sampler)
    def observed_sampler(*positional, **keywords):
        data = positional[2]
        if keywords.get('z') is True and data.shape[0] == expected['full_generation_windows']:
            path = runtime / ('full_sampler_inputs_' + str(len(sampler_inputs) + 1) + '.npz')
            states = {('cuda_rng_' + str(i)): state.cpu().numpy() for i, state in enumerate(torch.cuda.get_rng_state_all())}
            np.savez_compressed(path, latent_noise=data.detach().cpu().numpy(),
                conditioning_times=positional[3].detach().cpu().numpy(), torch_cpu_rng=torch.get_rng_state().numpy(), **states)
            sampler_inputs.append(dict(path=str(path), sha256=sha(path),
                native_generator_training_mode=bool(positional[1].training), full_windows=expected['full_generation_windows']))
        return native_sampler(*positional, **keywords)

    module.run_model = observed_sampler
    for name, role in [('discriminative_score_metrics', 'discriminative'),
            (expected['predictive_function'], 'predictive')]:
        original = getattr(module, name)

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

        setattr(module, name, bind(original, name, role))
    write_progress()
    previous_cwd = Path.cwd()
    try:
        # Original published Energy weight paths are relative to the author tree.
        os.chdir(runtime)
        module.main()
    finally:
        torch.optim.Adam = raw_adam
        os.chdir(previous_cwd)
    if len(created) != expected['expected_native_optimizers'] or counts['embedding'] != expected['embedding_optimizer_updates'] or counts['generator'] != expected['generator_optimizer_updates']:
        raise ValueError('Full original optimizer budgets did not complete')
    if not 0 <= counts['discriminator'] <= expected['maximum_discriminator_updates']:
        raise ValueError('Original conditional discriminator budget differs')
    if len(evaluations) != expected['full_periodic_evaluations']:
        raise ValueError('Every full original periodic evaluator required')
    if len(metric_receipts) != 20 * expected['full_periodic_evaluations']:
        raise ValueError('Every original metric evaluator must finish its full optimizer budget')
    checkpoints = []
    for path in sorted((runtime / 'native_run_checkpoints').glob('*.pt')):
        checkpoints.append(dict(path=str(path), sha256=sha(path), size=path.stat().st_size))
    if len(checkpoints) != 4 * expected['full_periodic_evaluations']:
        if not expected['native_release_checkpoint_evaluation'] or checkpoints:
            raise ValueError('All four native network checkpoints per training evaluation required')
    released_checkpoints = []
    if expected['native_release_checkpoint_evaluation']:
        sub = 'energy' if case['missing_rate'] == 0 else 'energy_0.5'
        for name in ['generator', 'recovery']:
            path = runtime / 'energy_model' / sub / (name + '.pt')
            released_checkpoints.append(dict(path=str(path), sha256=sha(path), size=path.stat().st_size))
    result = dict(captured_utc=datetime.now(timezone.utc).isoformat(), job=job, queue_sha256=sha(ROOT / args.queue),
                  construction_proof_sha256=sha(ROOT / queue['construction_proof']),
                  case=case, expected_budget=expected, optimizer_updates=counts, evaluations=evaluations,
                  checkpoints=checkpoints, released_checkpoints=released_checkpoints,
                  full_native_command_completed=True, training_seeds=0 if expected['native_release_checkpoint_evaluation'] else 1,
                  full_native_generator_training_completed=not expected['native_release_checkpoint_evaluation'],
                  author_equivalence_certified=False, strict_TAB_result=False,
                  native_no_pretrained_training_reload=True,
                  boundary=queue['boundary'])
    phase[0] = 'completed_full_native_command'
    write_progress()
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
