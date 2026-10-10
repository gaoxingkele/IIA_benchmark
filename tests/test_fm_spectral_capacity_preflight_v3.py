import pytest
from scripts.flow_matching import run_spectral_table2_spectral_preflight_v3 as worker
from scripts.flow_matching import run_registered_full_gpu_command_v2 as controller


def test_all_three_original_spectral_pipelines_selected():
    original={(method,dataset):[object()] for method in ('spectral_flow','sdformer_ar') for dataset in ('sine','stock','mujoco')}
    selected=worker.selected_pipelines(original)
    assert set(selected)=={('spectral_flow',d) for d in ('sine','stock','mujoco')}
    assert all(selected[k] is original[k] for k in selected)
    original.pop(('spectral_flow','mujoco'))
    with pytest.raises(ValueError,match='All three'):worker.selected_pipelines(original)


def test_frozen_full_forward_backward_optimizer_and_loader_function_is_preserved():
    full=worker.full_spectral_function()
    assert full.__globals__['original_array'] is worker.original.original_array
    assert full.__globals__['arguments_for'] is worker.original.arguments_for
    assert full.__globals__['verify_environment'] is worker.original.verify_environment
    def flatten(values):
        for value in values:
            if isinstance(value,tuple):yield from flatten(value)
            else:yield value
    constants=list(flatten(full.__code__.co_consts))
    assert 'backward_and_optimizer_step' in constants
    assert 'full_original_eight_worker_loaders' in constants
    assert {'backward','step','DATALoader','SpectralFlow'}.issubset(full.__code__.co_names)
    assert 512 in full.__code__.co_consts


def config():
    return dict(source_receipts=[],settings=dict(minimum_available_memory_bytes=32*1024**3,
        minimum_commit_headroom_bytes=42*1024**3,minimum_gpu_free_bytes=14*1024**3,
        emergency_commit_headroom_bytes=16*1024**3,emergency_gpu_free_bytes=4*1024**3))


@pytest.mark.parametrize('field,value', [('minimum_available_memory_bytes',31*1024**3),
    ('minimum_commit_headroom_bytes',41*1024**3),('minimum_gpu_free_bytes',13*1024**3),
    ('emergency_commit_headroom_bytes',10*1024**3),('emergency_gpu_free_bytes',2*1024**3)])
def test_conservative_capacity_guards_cannot_be_weakened(field,value):
    registration=config();registration['settings'][field]=value
    with pytest.raises(ValueError,match='admission gates'):controller.verify_registration(registration)


def test_complete_conservative_registration_is_accepted():
    controller.verify_registration(config())
