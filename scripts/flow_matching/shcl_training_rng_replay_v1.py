"""Restore every saved training RNG/backend state in a fresh read-only worker."""
import random
from pathlib import Path

from scripts.flow_matching import replay_shcl_native_frames_v1 as original
from scripts.flow_matching.shcl_native_observer_v1 import Observer


def verify_schema(checkpoint):
    if set(checkpoint['original_rng']) != {'python', 'numpy', 'torch_cpu', 'torch_cuda'}:
        raise ValueError('All four original training RNG families required')
    if set(checkpoint['backend_flags']) != {'cudnn_benchmark', 'cudnn_deterministic', 'cudnn_enabled',
            'cudnn_allow_tf32', 'matmul_allow_tf32', 'deterministic_algorithms'}:
        raise ValueError('Every original training backend flag required')
    if any(type(value) is not bool for value in checkpoint['backend_flags'].values()):
        raise ValueError('Exact original Boolean backend settings required')


def restore_and_verify_draws(checkpoint):
    import numpy as np
    import torch
    verify_schema(checkpoint)
    state = checkpoint['original_rng']
    if state['torch_cuda']:
        if not torch.cuda.is_available() or torch.cuda.device_count() != len(state['torch_cuda']):
            raise ValueError('Every original visible CUDA generator required; no CPU fallback')
        torch.cuda.get_rng_state_all()
    observation = Observer({}, '.')
    caller_rng = observation.rng()
    caller_flags = observation.backend_flags()
    try:
        original.set_backend_flags(checkpoint['backend_flags'])
        if observation.backend_flags() != checkpoint['backend_flags']:
            raise ValueError('All original backend flags must restore exactly')
        original.restore_rng(state)
        original.verify_rng(state)
        python_generator = random.Random()
        python_generator.setstate(state['python'])
        expected_python = [python_generator.getrandbits(64) for _ in range(16)]
        actual_python = [random.getrandbits(64) for _ in range(16)]
        numpy_generator = np.random.RandomState()
        numpy_generator.set_state(state['numpy'])
        expected_numpy = numpy_generator.random_sample(16)
        actual_numpy = np.random.random_sample(16)
        cpu_generator = torch.Generator(device='cpu')
        cpu_generator.set_state(state['torch_cpu'])
        expected_cpu = torch.rand(16, generator=cpu_generator)
        actual_cpu = torch.rand(16)
        if expected_python != actual_python or not np.array_equal(expected_numpy, actual_numpy) or not torch.equal(expected_cpu, actual_cpu):
            raise ValueError('Independent complete Python/NumPy/native CPU stream draws differ')
        for index, saved in enumerate(state['torch_cuda']):
            device = 'cuda:' + str(index)
            generator = torch.Generator(device=device)
            generator.set_state(saved)
            expected_cuda = torch.rand(16, generator=generator, device=device)
            actual_cuda = torch.rand(16, device=device)
            if not torch.equal(expected_cuda, actual_cuda):
                raise ValueError('Independent native CUDA stream draws differ')
        original.restore_rng(state)
        original.verify_rng(state)
        return dict(all_original_training_RNG_states_bitwise_restored=True,
            all_original_training_backend_flags_exactly_restored=True,
            independent_native_stream_draws_bitwise_verified=True,
            original_visible_CUDA_generators_verified=len(state['torch_cuda']),
            original_state_restored_after_diagnostic_draws=True, new_optimizer_updates_executed=0)
    finally:
        original.set_backend_flags(caller_flags)
        original.restore_rng(caller_rng)
        original.verify_rng(caller_rng)
        if observation.backend_flags() != caller_flags:
            raise ValueError('Read-only audit must preserve all caller backend settings')


def replay_training(model, training, checkpoint_path, report):
    import torch
    original.replay_training(model, training, checkpoint_path, report)
    restored = []
    for index, receipt in enumerate(training['epochs']):
        path = Path(receipt['full_state'])
        if original.native.sha(path) != receipt['full_state_sha256']:
            raise ValueError('Every original saved full training state must remain unchanged')
        checkpoint = torch.load(path, map_location='cpu')
        if checkpoint['source_job'] != training['job'] or checkpoint['completed_epoch'] != index + 1:
            raise ValueError('Exact original training job and epoch identity required')
        result = restore_and_verify_draws(checkpoint)
        restored.append(dict(epoch=index + 1, checkpoint=str(path), checkpoint_sha256=receipt['full_state_sha256'], **result))
        del checkpoint
    report.update(training_rng_checkpoints=restored, every_original_training_RNG_and_backend_state_verified=True,
        all_original_training_epoch_model_Adam_RNG_backend_restore_verified=True,
        synthetic_operator_only_not_benchmark_performance=training['job'].get('synthetic_fixture', False),
        new_optimizer_updates_executed=0, new_benchmark_results=0, original_paper_equivalence_certified=False,
        all_original_experiments_complete=False)
    return report
