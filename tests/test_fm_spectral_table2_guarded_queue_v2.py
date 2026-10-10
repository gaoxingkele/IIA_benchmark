"""The CPU wait binding must be available before any preflight is dispatched."""
import subprocess
import sys

from scripts.flow_matching.run_spectral_table2_guarded_queue_v2 import complete_resource_api
from scripts.flow_matching.run_spectral_table2_guarded_queue_v3 import bind_resource_api


def test_added_guard_handles_real_child_without_changing_other_bindings():
    original_gpu_guard=object()
    api=complete_resource_api(lambda:{'guarded_gpu_wait':original_gpu_guard})
    assert api['guarded_gpu_wait'] is original_gpu_guard
    process=subprocess.Popen([sys.executable,'-c','pass'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    exitcode,receipt=api['guarded_wait'](process,{'emergency_commit_headroom_bytes':0},poll_seconds=.01)
    assert exitcode==0 and not receipt['resource_guard_aborted']


def test_gpu_identity_is_added_without_mutating_frozen_scientific_settings():
    settings={'gpu_index':0,'minimum_gpu_free_bytes':14}
    api=bind_resource_api(lambda:{'gpu_snapshot':lambda supplied:dict(supplied)},'bound-uuid')
    actual=api['gpu_snapshot'](settings)
    assert actual==dict(settings,gpu_uuid='bound-uuid')
    assert 'gpu_uuid' not in settings and callable(api['guarded_wait'])
