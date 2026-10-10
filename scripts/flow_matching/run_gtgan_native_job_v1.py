"""Execute the original GT-GAN stock command without reducing its native budget."""
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

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def full_budget(case):
    parsed = case['parsed_original_arguments']
    regular = case['missing_rate'] == 0
    joint = 8500 if regular else 2000
    if (parsed['first_epoch'] != 10000 or parsed['max_steps'] != joint or
            parsed['batch_size'] != 128 or parsed['seq_len'] != 24 or
            parsed['max_steps_metric'] != 10 or parsed['solver'] != 'sym12async' or
            parsed['dims'] != '32-64-64-32' or case['original_seed'] != 7777 or
            parsed['log_time'] != (2 if regular else 1)):
        raise ValueError('Complete original GT-GAN command budget is required')
    return dict(native_embedding_steps=0 if regular else 10000, native_joint_steps=joint,
                embedding_optimizer_updates=(0 if regular else 10000) + 2 * joint,
                generator_optimizer_updates=joint + joint // parsed['log_time'],
                maximum_discriminator_updates=2 * joint, full_periodic_evaluations=joint // 500,
                evaluator_repeats_per_metric=10, full_generation_windows=3662, variables=6, sequence_length=24)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    queue = json.loads((ROOT / args.queue).read_text(encoding='utf-8'))
    for item in queue['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen GT-GAN source changed: ' + item['path'])
    lock = json.loads((ROOT / queue['environment_lock']).read_text(encoding='utf-8'))
    if sys.version != lock['python'] or any(importlib.metadata.version(k) != v for k, v in lock['packages'].items()):
        raise ValueError('Frozen original-version GT-GAN environment differs')
    construction = json.loads((ROOT / queue['construction_proof']).read_text(encoding='utf-8'))
    if construction['full_original_constructor_cases'] != 4 or construction['performance_result']:
        raise ValueError('All four native constructor/weight checks required')
    job = next(j for j in queue['jobs'] if j['id'] == args.job_id)
    model = json.loads((ROOT / queue['model_config']).read_text(encoding='utf-8'))
    case = next(c for c in model['cases'] if c['id'] == job['case_id'])
    expected = full_budget(case)
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
    spec = importlib.util.spec_from_file_location('native_gtgan_training', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    phase = ['native_initialization']
    counts = dict(embedding=0, generator=0, discriminator=0)
    created = []
    evaluations = []
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
            step = local.get('step')
            if step != 500 * (len(evaluations) + 1):
                raise ValueError('Complete original periodic evaluator cadence differs')
            real = np.asarray(local['generated_data1'])
            generated = np.asarray(local['generated_data2'])
            if real.shape != (3662, 24, 6) or generated.shape != real.shape or not np.isfinite(generated).all():
                raise ValueError('Full native generation window/variable coverage required')
            if len(local['predictive_score']) != 10 or len(local['discriminative_score']) != 10:
                raise ValueError('Ten full original evaluator repetitions required')
            artifact = runtime / ('full_generation_step_' + str(step) + '.npz')
            with artifact.open('xb') as stream:
                np.savez_compressed(stream, real=real, generated=generated)
            row = dict(native_joint_step=step, original_metrics={k: float(v) for k, v in values[0].items()},
                       predictive_repeats=[float(v) for v in local['predictive_score']],
                       discriminative_repeats=[float(v) for v in local['discriminative_score']],
                       full_generation_windows=3662, artifact_path=str(artifact), artifact_sha256=sha(artifact),
                       uncertainty_scope='Evaluator repeats of one trained model, not independent training seeds')
            evaluations.append(row)
            (output / ('evaluation_' + str(step) + '.json')).write_text(json.dumps(row, indent=2) + '\n', encoding='utf-8')
            write_progress()
        raw_print(*values, **kwargs)

    sys.argv = [str(file)] + case['author_arguments'] + ['--save_dir', 'native_run_checkpoints']
    torch.optim.Adam = ObservedAdam
    module.optim.Adam = ObservedAdam
    module.__dict__['print'] = observed_print
    write_progress()
    try:
        module.main()
    finally:
        torch.optim.Adam = raw_adam
    if len(created) != 3 or counts['embedding'] != expected['embedding_optimizer_updates'] or counts['generator'] != expected['generator_optimizer_updates']:
        raise ValueError('Full original optimizer budgets did not complete')
    if not 0 <= counts['discriminator'] <= expected['maximum_discriminator_updates']:
        raise ValueError('Original conditional discriminator budget differs')
    if len(evaluations) != expected['full_periodic_evaluations']:
        raise ValueError('Every full original periodic evaluator required')
    checkpoints = []
    for path in sorted((runtime / 'native_run_checkpoints').glob('*.pt')):
        checkpoints.append(dict(path=str(path), sha256=sha(path), size=path.stat().st_size))
    if len(checkpoints) != 4 * expected['full_periodic_evaluations']:
        raise ValueError('All four native network checkpoints per evaluation required')
    result = dict(captured_utc=datetime.now(timezone.utc).isoformat(), job=job, queue_sha256=sha(ROOT / args.queue),
                  case=case, expected_budget=expected, optimizer_updates=counts, evaluations=evaluations,
                  checkpoints=checkpoints, full_native_command_completed=True, training_seeds=1,
                  author_equivalence_certified=False, strict_TAB_result=False,
                  native_released_pretrain_semantics_preserved=True,
                  boundary=queue['boundary'])
    phase[0] = 'completed_full_native_command'
    write_progress()
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
