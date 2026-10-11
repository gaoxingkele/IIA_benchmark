"""Full50-epoch TCCM worker after an actual TCCM batch32 capacity proof."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read, sha, verify_receipts
from scripts.flow_matching.full_image_training_execution_recovery_v2 import (
    enable_native_activation_recomputation, native_path, save_checkpoint, write_json)
from scripts.flow_matching.run_full_image_training_execution_recovery_v3 import training_ast


def verify_capacity(config):
    cases = config['capacity_cases']
    if len(cases) != 1 or cases[0]['objective'] != 'TCCM':
        raise ValueError('Native TCCM capacity cannot be replaced by FM or WT')
    location = read(ROOT / cases[0]['output_directory'] / 'capacity_probe_result.json')
    verify_receipts([location])
    proof = read(ROOT / location['path'])
    terminal_path = Path(location['path']).parent / 'terminal.json'
    terminal = read(terminal_path)
    if (proof['case'] != cases[0] or proof['configuration_sha256'] != sha(ROOT / config['configuration_path'])
            or proof['actual_batch_shape'] != [32, 3, 128, 128]
            or proof['full_model_parameters'] != 45289539
            or proof['native_recomputed_blocks'] != {'ResBlock': 54, 'AttentionBlock': 29}
            or not proof['full_batch_forward_backward_optimizer_and_checkpoint_IO_passed']
            or proof['actual_optimizer_updates'] != 1 or not proof['capacity_probe_only']
            or proof['full_original_training_executed'] or proof['benchmark_performance']
            or not proof['bfloat16'] or not proof['TF32'] or terminal['returncode'] != 0
            or terminal['resource_usage']['resource_guard_aborted']):
        raise ValueError('Successful actual full TCCM CUDA/BF16/TF32 update and checkpoint required')
    verify_receipts([proof['checkpoint']])
    return dict(proof=location, terminal=dict(path=str(terminal_path), sha256=sha(terminal_path)),
        capacity_probe_weights_loaded=False)


def run(config_path, job_path, attempt_directory):
    config = read(ROOT / config_path); verify_receipts(config['source_receipts'])
    proof = verify_capacity(config)
    queue = read(ROOT / config['original_training_queue'])
    if job_path not in [entry['path'] for entry in queue['jobs']] or len(queue['jobs']) != 5:
        raise ValueError('Only the five unchanged full50epoch TCCM jobs may run')
    from scripts.flow_matching import run_tccm_full_image_training_v1 as original
    job = read(ROOT / job_path)
    attempt = Path(attempt_directory).resolve()
    if not attempt.is_relative_to((ROOT / job['output_directory']).resolve() / 'attempts') or not attempt.is_dir():
        raise ValueError('Fresh original canonical attempt required')

    def hook(model, model_config, environment, actual_job, attempt_directory):
        if actual_job != job:
            raise ValueError('Original full TCCM job changed')
        expected = next(x for x in environment['full_architectures']
            if x['dataset'] == job['dataset'] and x['variant'] == job['variant'])
        count = sum(p.numel() for p in model.parameters())
        if count != 45289539 or model_config['parameters'] != expected['architecture_parameters']:
            raise ValueError('Original complete TCCM backbone required')
        blocks = enable_native_activation_recomputation(model)
        if blocks != {'ResBlock': 54, 'AttentionBlock': 29}:
            raise ValueError('All native full-backbone blocks required')
        write_json(attempt / 'verified_full_tccm_execution_adapter.json', dict(
            original_job=job_path, original_job_sha256=sha(ROOT / job_path),
            full_parameter_count=count, native_recomputed_blocks=blocks,
            complete_original_epochs=50, batch_size=32, bfloat16=True, TF32=True,
            original_dataset_images_per_epoch=3629, original_total_optimizer_updates=5700,
            native_checkpoint_mode='nonreentrant_preserve_rng_state',
            original_full_train_epoch_and_TCCM_loss_functions_unchanged=True,
            native_capacity_proof_binding=proof, capacity_probe_weights_loaded=False,
            benchmark_performance=False, original_author_equivalence_certified=False))

    proxy = SimpleNamespace(**vars(original.os))
    proxy.link = lambda source, destination: original.os.link(native_path(source), native_path(destination))
    namespace = dict(vars(original), write=write_json, save_checkpoint=save_checkpoint,
        os=proxy, _verify_execution_hook=hook)
    source = Path(original.__file__)
    exec(compile(training_ast(source), str(source), 'exec'), namespace)
    sys.argv = [str(source), '--config', job_path, '--attempt-directory', str(attempt)]
    namespace['main']()
    verify_receipts(config['source_receipts'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery-config', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--attempt-directory', required=True)
    args = parser.parse_args()
    run(args.recovery_config, args.config, args.attempt_directory)
