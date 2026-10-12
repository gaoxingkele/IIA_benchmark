"""Replace a verified idle host monitor; frozen native workers remain unchanged.

Only the resource_snapshot binding in freshly compiled orchestration namespaces
uses the cached Windows type. No live process or frozen source is hot patched.
Original model dispatch, data, artifact validators, caps and budgets are retained.
"""
import argparse
from datetime import datetime, timezone
import importlib
import json
import os
import sys

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen monitor handoff source changed: ' + pin['path'])
    if (config['entry_module'] != 'scripts.flow_matching.run_rcgan_native_original_queue_v2'
            or config['original_scope'] != dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273)):
        raise ValueError('Only explicitly registered original full500 RCGAN monitor supported')
    queue = read(ROOT / config['original_queue'])
    from scripts.flow_matching.run_rcgan_native_original_queue_v2 import verify_registration
    verify_registration(queue)
    if config['settings'] != queue['settings'] or config['resource_lock'] != queue['resource_lock']:
        raise ValueError('All original full training limits and family locks retained')
    return queue


def install_cached_loader(first_snapshot):
    from scripts.flow_matching import run_light_controller as loader
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    original_loader = loader.load_functions

    def cached_snapshot():
        result = resource_snapshot()
        first_snapshot(result)
        return result

    def load_functions(root, path, names, namespace):
        functions = original_loader(root, path, names, namespace)
        if (root / path).resolve() == (ROOT / 'scripts/flow_matching/heavy_resources.py').resolve() and 'resource_snapshot' in names:
            namespace['resource_snapshot'] = cached_snapshot
            functions['resource_snapshot'] = cached_snapshot
        return functions

    loader.load_functions = load_functions
    # verify() may have imported a client before installation; explicitly bind
    # its future namespace compilation to the same isolated fresh-process loader.
    from scripts.flow_matching import run_cfm_ts_low_memory_queue_v2 as client
    client.load_functions = load_functions
    return load_functions


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--runtime-config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = read(ROOT / args.runtime_config)
    queue = verify(config)
    if args.probe_only:
        print(json.dumps(dict(passed=True, original_seed_jobs=5, original_epochs_each=500,
            original_guards_unchanged=True, source_dispatch_unchanged=True, training_executed=False)))
        return
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.run_timegan_corrected_observer_dispatch_v2 import safe_handoff
    base = ROOT / config['monitor_state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = resource_api()['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Same cached host monitor already owned')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), state='starting', active_pid=None,
        configuration_sha256=sha(ROOT / args.runtime_config), all_original_experiments_complete=False)

    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat())
        write(base / 'status.json', state)

    recorded = [False]

    def first_snapshot(result):
        if not recorded[0]:
            write(base / 'actual_first_cached_snapshot.json', dict(snapshot=result, pid=os.getpid(),
                create_time=psutil.Process().create_time(), settings=queue['settings'],
                original_snapshot_formula_and_Windows_API_unchanged=True,
                cached_snapshot_sha256=sha(ROOT / 'scripts/flow_matching/cached_resource_snapshot_v1.py')))
            recorded[0] = True

    try:
        update({})
        safe_handoff('original', config['handoff'], base, update, 10)
        install_cached_loader(first_snapshot)
        update(dict(state='running_original_controller_with_cached_resource_telemetry'))
        sys.argv = ['original_full500_native_controller', '--queue', config['original_queue']]
        importlib.import_module(config['entry_module']).main()
        update(dict(state='original_full500_controller_returned'))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
