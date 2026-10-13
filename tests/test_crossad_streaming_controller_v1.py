"""Checksum equivalence and exact idle-only cold controller handoff gates."""
import hashlib

import pytest

from scripts.flow_matching import run_crossad_streaming_controller_v1 as cold


def test_streaming_hash_covers_full_nonmultiple_block_file(tmp_path):
    path = tmp_path / 'all_bytes.bin'
    payload = bytes(range(256)) * 16385 + b'last trailing original bytes'
    path.write_bytes(payload)
    assert cold.original.digest(path) == hashlib.sha256(payload).hexdigest()
    with path.open('ab') as stream: stream.write(b'changed')
    assert cold.original.digest(path) != hashlib.sha256(payload).hexdigest()


def idle():
    return dict(state='waiting_for_heavy_lock', active_pid=None, active_job=None,
        active_case=None, active_stage=None, shared_lock_owned=False)


@pytest.mark.parametrize('field,value', [('active_pid', 1), ('active_job', 'running'),
    ('active_case', 'running'), ('active_stage', 'training'), ('shared_lock_owned', True),
    ('state', 'running')])
def test_any_native_work_or_reservation_blocks_handoff(field, value):
    state = idle(); state[field] = value
    assert not cold.idle_allowed('original', state, False, [])


def test_native_descendant_blocks_even_when_status_claims_idle():
    assert not cold.idle_allowed('original', idle(), False,
        [dict(name='python.exe', exe='C:/python.exe')])
    assert cold.idle_allowed('original', idle(), False, [])


def test_original_complete_controller_code_and_canonical_queue_preserved():
    assert cold.bound_main().__code__ is cold.original.main.__code__
    runtime = cold.json.loads((cold.ROOT / 'configs/runtime/fm_light_controllers.v1.json').read_text(encoding='utf-8'))
    controller, namespace, binding = cold.optimized_load(cold.ROOT, runtime, 'crossad')
    assert namespace['sha'] is cold.original.digest
    assert namespace['resource_snapshot'] is cold.resource_snapshot
    assert controller.main.__globals__ is namespace
    assert binding['queue_path'] == 'configs/experiments/fm_crossad_complete_queue.v7.json'
