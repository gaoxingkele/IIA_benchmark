"""Retain the exact full9 industrial controller in its cached module namespace."""
import argparse
import importlib
import json
import os
import sys

import psutil

from scripts.flow_matching import run_idle_exact_command_cached_handoff_v3 as prior


def prepare_entry():
    # runpy would re-execute run_light_controller.py as __main__, defining a new
    # uncached load_functions. Call the original main in its installed namespace.
    prior.install_cached_loader(lambda snapshot: None)
    module = importlib.import_module('scripts.flow_matching.run_light_controller')
    return module.main


def verify(config):
    if config['original_scope'] != prior.SCOPE or config['original_role'] != 'industrial9':
        raise ValueError('Retain the complete original industrial9 and global scope')
    for pin in config['source_receipts']:
        if prior.digest(prior.ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen industrial namespace handoff changed: ' + pin['path'])
    registration = prior.read(prior.ROOT / config['original_registration'])
    for name in prior.ENTRIES:
        prior.verify(registration, name)
    original = registration['roles']['industrial9']
    if config['original_arguments'] != original['original_arguments']:
        raise ValueError('Every original industrial argument must remain exact')
    command = config['previous_owner']['command']
    index = command.index('-m')
    if (command[index + 1] != 'scripts.flow_matching.run_idle_exact_command_cached_handoff_v3'
            or command[index + 2:] != ['--config', config['original_registration'], '--role', 'industrial9']):
        raise ValueError('Only the exact registered idle industrial v3 successor may be handed off')
    if config['previous_state_path'] != original['previous_state_path']:
        raise ValueError('The same original full industrial state is required')
    return original


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = prior.read(prior.ROOT / args.config)
    original = verify(config)
    entry = prepare_entry()
    if args.probe_only:
        from scripts.flow_matching.run_light_controller import load_controller
        from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
        runtime = prior.read(prior.ROOT / original['original_runtime'])
        _, namespace, binding = load_controller(prior.ROOT, runtime, 'industrial_empty_output_recovery')
        snapshot = namespace['resource_snapshot']
        assert snapshot.__name__ == 'cached_snapshot'
        print(json.dumps(dict(passed=True, complete_original_slots=9,
            original_main_called_in_cached_module_namespace=True,
            actual_loaded_controller_resource_snapshot_name=snapshot.__name__,
            original_queue_sha256=binding['queue_sha256'], snapshot=resource_snapshot(),
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules,
            private_bytes=psutil.Process().memory_info().private,
            numerical_training_executed=False)))
        return
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    base = prior.ROOT / config['handoff_state_root']
    base.mkdir(parents=True, exist_ok=True)
    owner = resource_api()['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Industrial cached namespace successor already owned')
    try:
        prior.handoff(config, base)
        verify(config)
        prior.exclusive(base / 'successor_entry.json', dict(pid=os.getpid(),
            original_entry=original['entry_module'], original_arguments=config['original_arguments'],
            original_complete_slots=9, original_scope=prior.SCOPE,
            original_main_called_in_cached_module_namespace=True))
        sys.argv = [original['entry_module'], *config['original_arguments']]
        entry()
    finally:
        owner.close()


if __name__ == '__main__':
    main()
