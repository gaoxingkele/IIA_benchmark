"""Recover the stopped metadata collector and retain all45 native replay slots.

The full original Sine numerical worker and every guard remain unchanged.
Host hashes use a reusable64KiB buffer. The actual Stock1104 saved-graph proof
joins the existing certificates, without running its completed experiment again.
TimeGAN: Yoon, Jarrett, van der Schaar, NeurIPS2019.
"""
import ast
import hashlib
import inspect
from pathlib import Path
import sys
from types import FunctionType

from scripts.flow_matching import timegan_saved_joint_full_replay_v1 as original

BASE, ROOT = original.BASE, original.ROOT
PREVIOUS_SUPPLEMENT = 'configs/runtime/fm_timegan_sine_saved_joint15000_full_replay.v1.json'
STOCK = 'configs/runtime/fm_timegan_stock1104_saved_graph_full_replay.v1.json'
CANONICAL = original.CANONICAL
HASH_BUFFER_BYTES = 64 * 1024


def bounded_sha(path):
    digest = hashlib.sha256()
    buffer = bytearray(HASH_BUFFER_BYTES)
    view = memoryview(buffer)
    with Path(path).open('rb', buffering=0) as stream:
        while True:
            count = stream.readinto(buffer)
            if not count:
                break
            digest.update(view[:count])
    return digest.hexdigest()


def verify(config):
    previous = BASE.read(ROOT / PREVIOUS_SUPPLEMENT)
    changed = {key for key in set(config) | set(previous) if config.get(key) != previous.get(key)}
    allowed = {'runtime_configuration', 'state_root', 'output_root', 'source_receipts',
               'previous_supplement_configuration', 'supplemental_stock_configuration',
               'stopped_parent_incident', 'host_hash_buffer_bytes'}
    if not changed <= allowed or config['routes'] != previous['routes']:
        raise ValueError('Only isolated host collector/output/evidence fields may change')
    if (config['previous_supplement_configuration'] != PREVIOUS_SUPPLEMENT
            or config['supplemental_stock_configuration'] != STOCK
            or config['host_hash_buffer_bytes'] != HASH_BUFFER_BYTES
            or config['state_root'] == previous['state_root']
            or config['output_root'] == previous['output_root']):
        raise ValueError('Distinct immutable outputs and bounded hash mechanism required')
    pins = {p['path']: p['sha256'] for p in config['source_receipts']}
    if len(pins) != len(config['source_receipts']):
        raise ValueError('Duplicate source pins')
    for pin in previous['source_receipts']:
        if pins.get(pin['path']) != pin['sha256']:
            raise ValueError('Every original frozen numerical/source pin must remain')
    required = [PREVIOUS_SUPPLEMENT, STOCK, 'scripts/flow_matching/timegan_saved_joint_full_replay_v2.py',
                'tests/test_timegan_saved_joint_full_replay_v2.py', config['stopped_parent_incident']]
    if any(pins.get(p) != bounded_sha(ROOT / p) for p in required):
        raise ValueError('New immutable collector/incident/evidence pins required')
    incident = BASE.read(ROOT / config['stopped_parent_incident'])
    if (incident['failed_before_native_child'] is not True
            or incident['scientific_worker_repeated'] is not False
            or incident['failure_type'] != 'MemoryError'
            or incident['native_replay_started'] is not False):
        raise ValueError('Preserved actual host-only failure required')
    return original.verify(config)


def collect_full45_evidence(config):
    # Reuse the original certificate verifier bytecode. Only its host SHA
    # implementation changes; all hashes still cover every byte of every file.
    collector = FunctionType(original.collect_full45_evidence.__code__,
                             dict(original.collect_full45_evidence.__globals__, sha=bounded_sha))
    primary = collector(config)
    from scripts.flow_matching import timegan_stock1104_saved_graph_full_replay_v1 as stock
    stock_config = BASE.read(ROOT / STOCK)
    extra_collector = FunctionType(original.collect_full45_evidence.__code__,
        dict(original.collect_full45_evidence.__globals__, sha=bounded_sha, CANONICAL=stock.CANONICAL))
    extra = extra_collector(stock_config)
    routes = {r['canonical_job']['id']: r for r in config['routes']}
    for identity, proof in extra['verified_native_evidence'].items():
        if (proof['canonical_job'] != routes[identity]['canonical_job']
                or proof['execution_job'] != routes[identity]['execution_job']):
            raise ValueError('Supplemental certificate is not the same canonical/execution slot')
        existing = primary['verified_native_evidence'].get(identity)
        if existing is not None and existing != proof:
            raise ValueError('Conflicting full native proof for one canonical seed')
        primary['verified_native_evidence'][identity] = proof
    primary['verified_original_ids'] = sorted(primary['verified_native_evidence'])
    primary['pending_original_models'] = 45 - len(primary['verified_original_ids'])
    primary['all45_TimeGAN_native_replays_complete'] = primary['pending_original_models'] == 0
    return primary


def native_namespace():
    namespace = original.native_namespace()
    namespace['verify'] = verify
    # No native model, array, optimizer, checkpoint, seed or bitwise gate changes.
    return namespace


def main_ast():
    tree = ast.parse(inspect.getsource(original.main))
    legacy = sys.version_info < (3, 8)
    literal = ast.Str if legacy else ast.Constant
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, literal):
            continue
        value = node.s if legacy else node.value
        old = "runpy.run_module('scripts.flow_matching.timegan_saved_joint_full_replay_v1'"
        if isinstance(value, str) and old in value:
            updated = value.replace(old, "runpy.run_module('scripts.flow_matching.timegan_saved_joint_full_replay_v2'")
            if legacy:
                node.s = updated
            else:
                node.value = updated
            count += 1
    if count != 1:
        raise ValueError('Only the isolated native bootstrap module may change')
    return ast.fix_missing_locations(tree)


def retired_owner_absent(config):
    import psutil
    incident = BASE.read(ROOT / config['stopped_parent_incident'])
    owner = incident['previous_owner']
    try:
        process = psutil.Process(owner['pid'])
        same_time = abs(process.create_time() - owner['create_time']) < .001
        same_command = process.cmdline() == owner['command']
    except psutil.NoSuchProcess:
        return
    if same_time or same_command:
        raise RuntimeError('Original collector still exists; no duplicate dispatcher permitted')


def main():
    # Verify the missing exact parent before taking a fresh owner lock. Never
    # stop a process or repeat an executed/failed numerical child.
    if '--worker' not in sys.argv and '--probe-only' not in sys.argv:
        index = sys.argv.index('--config')
        config = BASE.read(ROOT / sys.argv[index + 1])
        verify(config)
        retired_owner_absent(config)
    namespace = dict(original.main.__globals__, sha=bounded_sha, verify=verify,
                     collect_full45_evidence=collect_full45_evidence, native_namespace=native_namespace)
    exec(compile(main_ast(), __file__, 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
