"""Preserve v1 replay; recheck completed artifacts across source-exit races."""
import argparse
import sys
from types import FunctionType

import psutil

from scripts.flow_matching import run_pi_causal1107_deferred_full_replay_v1 as original
from scripts.flow_matching import run_idle_exact_command_cached_handoff_v3 as cold


def ready_or_exact_worker_live(config, queue):
    try:
        return original.ready_or_exact_worker_live(config, queue)
    except (RuntimeError, psutil.NoSuchProcess) as observation_error:
        # The exact worker can publish its terminal result and exit between the
        # first file read and the PID query. Terminal artifact evidence wins.
        try:
            original.registration.require_original_complete(queue)
        except (OSError, ValueError):
            raise observation_error
        return True


def verify(config):
    queue, job = original.verify(config)
    parent = original.registration.read(config['previous_watcher_configuration'])
    if (original.digest(original.ROOT / config['previous_watcher_configuration'])
            != config['previous_watcher_configuration_sha256']
            or parent['canonical_id'] != config['canonical_id']
            or parent['source_worker'] != config['source_worker']
            or parent['original_budget'] != config['original_budget']):
        raise ValueError('The same frozen v1 watcher and unchanged original source are required')
    command = config['previous_owner']['command']
    index = command.index('-m')
    if (command[index+1] != 'scripts.flow_matching.run_pi_causal1107_deferred_full_replay_v1'
            or command[index+2:] != ['--config', config['previous_watcher_configuration']]):
        raise ValueError('Only the exact idle v1 metadata watcher may be replaced')
    if config['previous_state_path'] != parent['state_root'] + '/status.json':
        raise ValueError('Exact original metadata state required')
    return queue, job


def idle_watcher(state, expected, descendants):
    if state.get('state') != 'waiting_exact_original_full_training_result':
        return False
    return (state.get('canonical_id') == original.replay.CANONICAL
            and cold.idle_allowed(dict(state, state='waiting_for_memory'), expected, descendants))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = original.registration.read(args.config)
    verify(config)
    values = dict(original.__dict__, verify=verify,
                  ready_or_exact_worker_live=ready_or_exact_worker_live)
    if not args.probe_only:
        from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader
        install_cached_loader(lambda snapshot: None)
        from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
        base = original.ROOT / config['state_root'] / 'idle_watcher_transition'
        base.mkdir(parents=True, exist_ok=True)
        owner = resource_api()['exclusive_lock'](base / 'handoff.lock')
        if owner is None:
            raise RuntimeError('The exact deferred-watcher transition is already owned')
        try:
            function = FunctionType(cold.handoff.__code__, dict(cold.__dict__, idle_allowed=idle_watcher),
                                    argdefs=cold.handoff.__defaults__)
            function(config, base)
        finally:
            owner.close()
    FunctionType(original.main.__code__, values)()


if __name__ == '__main__':
    main()
