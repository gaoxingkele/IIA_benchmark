"""Preserve all49 state audits while enforcing the original shared GPU mutex."""
import ast
import inspect

from scripts.flow_matching import run_shcl_training_rng_queue_v2 as previous
from scripts.flow_matching.combined_readonly_gpu_slot_v1 import combined_slot, verify_gpu_mutex
from scripts.flow_matching import run_cached_resource_monitor_controller_v2 as handoff
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main_ast():
    tree = ast.parse(inspect.getsource(previous.original.main))
    count = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'write'
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'native'):
            node.func = ast.Name(id='write', ctx=ast.Load())
        if isinstance(node, ast.With):
            for item in node.items:
                if isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Name) and item.context_expr.func.id == 'slot':
                    item.context_expr = ast.parse('combined_slot(api, config, update, fixture)', mode='eval').body
                    count += 1
    if count != 1:
        raise ValueError('Exactly one original read-only dispatch gains shared GPU reservation')
    return ast.fix_missing_locations(tree)


def safe_phase(config, state, descendants):
    old = config['handoff']['previous_owner']
    return (state.get('pid') == old['pid'] and state.get('create_time') == old['create_time']
        and state.get('state') == 'waiting_for_full_original_training_checkpoints' and state.get('active_pid') is None
        and state.get('required_count') == 49 and state.get('verified_count') == 0 and not state.get('failures')
        and state.get('configuration_sha256') == config['handoff']['previous_configuration_sha256']
        and all(handoff.allowed_console_child(p['name'], p['exe']) for p in descendants))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    native = previous.original.job_api.replay.original.native
    config = native.read(native.ROOT / args.config)
    previous.original.job_api.verify(config)
    verify_gpu_mutex(config)
    base = native.ROOT / config['state_root']
    base.mkdir(parents=True, exist_ok=True)
    from types import FunctionType
    stop = FunctionType(handoff.safe_handoff.__code__, dict(handoff.__dict__, safe_phase=safe_phase))
    stop(config, base, lambda values: previous.write(base / 'idle_handoff_wait.json', values))
    namespace = dict(previous.original.__dict__, write=previous.write, combined_slot=combined_slot)
    exec(compile(main_ast(), __file__, 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
