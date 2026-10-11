"""Wait for actual TCCM capacity, then dispatch all five unchanged full50epoch jobs."""
import argparse
import ast
from datetime import datetime, timezone
from pathlib import Path
import os
import sys
import time
import psutil

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read, sha, verify_receipts
from scripts.flow_matching.run_resource_gated_light_controller_v1 import write
from scripts.flow_matching.run_tccm_full_image_execution_recovery_v1 import verify_capacity
from scripts.flow_matching import run_tccm_full_image_training_queue_v1 as original


def queue_ast(source):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main']
    if len(tree.body) != 1:
        raise ValueError('Exactly one original full TCCM training queue required')
    changes = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'command'):
            expected = 'scripts.flow_matching.run_tccm_full_image_training_v1'
            if not isinstance(node.value, ast.List) or sum(isinstance(n, ast.Constant) and n.value == expected for n in node.value.elts) != 1:
                raise ValueError('Frozen native worker command changed')
            for n in node.value.elts:
                if isinstance(n, ast.Constant) and n.value == expected:
                    n.value = 'scripts.flow_matching.run_tccm_full_image_execution_recovery_v1'
            node.value.elts += [ast.Constant('--recovery-config'),
                ast.parse("runtime['recovery_configuration']", mode='eval').body]
            changes += 1
    if changes != 1:
        raise ValueError('Exactly one full native worker dispatch required')
    return ast.fix_missing_locations(tree)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    runtime = read(ROOT / args.runtime_config); verify_receipts(runtime['source_receipts'])
    config = read(ROOT / runtime['recovery_configuration']); verify_receipts(config['source_receipts'])
    namespace = dict(original.__dict__)
    source = Path(original.__file__)
    exec(compile(queue_ast(source.read_text(encoding='utf-8')), str(source), 'exec'), namespace)
    if args.validate_only:
        sys.argv = [str(source), '--runtime-config', args.runtime_config, '--validate-only']
        namespace['main'](); return
    process = psutil.Process()
    state_root = ROOT / runtime['state_root']; state_root.mkdir(parents=True, exist_ok=True)
    owner = original.queue_api()['exclusive_lock'](state_root / 'capacity_wait_controller.lock')
    if owner is None:
        raise RuntimeError('Existing full TCCM recovery owns the waiting lane')
    try:
        while not (ROOT / config['capacity_cases'][0]['output_directory'] / 'capacity_probe_result.json').exists():
            write(state_root / 'status.json', dict(pid=process.pid,
                create_time=process.create_time(), actual_command=process.cmdline(), active_pid=None,
                state='waiting_for_actual_full_native_TCCM_capacity', registered_jobs=5,
                complete=False, benchmark_performance=False, captured_utc=datetime.now(timezone.utc).isoformat()))
            time.sleep(10)
        proof = verify_capacity(config)
        write(state_root / 'actual_full_native_capacity_binding.json', proof)
        sys.argv = [str(source), '--runtime-config', args.runtime_config]
        namespace['main']()
    finally:
        owner.close()


if __name__ == '__main__':
    main()
