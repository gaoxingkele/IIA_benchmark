from types import SimpleNamespace
import sys
from scripts.flow_matching.run_light_controller import ROOT
from scripts.flow_matching.summarize_tsad_execution import light_worker_matches


def test_new_gpu_light_controller_is_bound_to_its_original_frozen_queue():
    process = SimpleNamespace(cmdline=lambda: [sys.executable, '-m', 'scripts.flow_matching.run_light_controller',
        '--runtime-config', 'configs/runtime/fm_tsad_gpu_light_controller.v1.json', '--controller', 'tsad_gpu'])
    assert light_worker_matches(process, 'configs/experiments/fm_tsad_gpu_queue.v1.json', ROOT)
    assert not light_worker_matches(process, 'configs/experiments/fm_tsad_cpu_queue.v1.json', ROOT)
    assert not light_worker_matches(process, 'configs/experiments/fm_giflow_native_queue.v3.json', ROOT)
