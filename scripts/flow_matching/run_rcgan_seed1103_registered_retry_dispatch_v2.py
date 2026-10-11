"""Dispatch the existing full262 retry with its exact registered native identity.

The former lightweight waiting owner is handed off separately, only when it has
no native child. Original checkpoints, completed epochs, failed attempts, all
five seeds, source implementations, admission and emergency caps are preserved.
"""
import argparse
import ast
import copy
import json
import os
from pathlib import Path
import sys

from scripts.flow_matching import run_rcgan_native_state_recovery_v1 as original
from scripts.flow_matching.run_preserved_native_gan_retry_v1 import controller_ast, admission_settings
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha


def is_windows_console_host(name, executable):
    expected = Path(os.environ.get('SystemRoot', 'C:/Windows'))/'System32/conhost.exe'
    return name.lower() == 'conhost.exe' and os.path.normcase(str(Path(executable).resolve())) == os.path.normcase(str(expected.resolve()))


def actual_native_children(process):
    return [child for child in process.children(recursive=True)
            if not is_windows_console_host(child.name(), child.exe())]


def dispatch_ast(source):
    tree = controller_ast(source, 'rcgan')
    counts = dict(state=0, imports=0, command=0)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'base'):
            if ast.dump(node.value) != ast.dump(ast.parse("ROOT/config['state_root']", mode='eval').body):
                raise ValueError('Frozen original state path binding changed')
            node.value = ast.parse("ROOT/_dispatch_configuration['state_root']", mode='eval').body
            counts['state'] += 1
        if isinstance(node, ast.ImportFrom) and node.module == 'scripts.flow_matching.rcgan_native_training_resume_runtime_v1':
            if len(node.names) != 1 or node.names[0].name not in ('run', 'verify_result'):
                raise ValueError('Only complete original native run/result verifier bindings may change')
            node.module = 'scripts.flow_matching.rcgan_seed1103_registered_retry_runtime_v2'
            counts['imports'] += 1
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'command'):
            if not isinstance(node.value, ast.List): raise ValueError('Exact original worker argv list required')
            module = [n for n in node.value.elts if isinstance(n, ast.Constant)
                      and n.value == 'scripts.flow_matching.run_rcgan_native_state_recovery_v1']
            if len(module) != 1: raise ValueError('One complete original native worker dispatch required')
            module[0].value = 'scripts.flow_matching.run_rcgan_seed1103_registered_retry_dispatch_v2'
            position = node.value.elts.index(module[0]) + 1
            node.value.elts[position:position] = [ast.Constant(value='--runtime-config'),
                ast.Name(id='_dispatch_configuration_path', ctx=ast.Load())]
            counts['command'] += 1
    if counts != dict(state=1, imports=2, command=1):
        raise ValueError('Only controller path and exact native adapter dispatch may differ: '+str(counts))
    return ast.fix_missing_locations(tree)


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path']) != pin['sha256']:
            raise ValueError('Frozen exact full retry dispatch changed: '+pin['path'])
    native = read(ROOT/config['native_configuration'])
    queue, state_config = original.verify_registration(native)
    if (sha(ROOT/config['native_configuration']) != config['native_configuration_sha256']
            or native['canonical_job']['id'] != 'rcgan_mnistfull__seed1103'
            or native['recovery_job']['id'] != 'rcgan_mnistfull__seed1103__state_recovery_v2'
            or native['verified_completed_epochs'] != 238 or native['remaining_full_epochs'] != 262
            or native['full_original_epochs'] != 500 or native['recovery_job']['seed'] != 1103
            or config['state_root'] == native['state_root']
            or config['settings'] != native['settings']
            or config['resource_lock'] != native['resource_lock']
            or config['original_scope'] != native['original_scope']
            or native['retry_admission_commit_headroom_bytes'] != 40*1024**3
            or len(queue['jobs']) != 5 or set(j['seed'] for j in queue['jobs']) != set(range(1103,1108))):
        raise ValueError('All original five500epoch obligations, exact238+262 recovery, and all guards required')
    return native, queue, state_config


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--runtime-config', required=True)
    args, remaining = parser.parse_known_args()
    config = read(ROOT/args.runtime_config)
    native, queue, state_config = verify(config)
    if '--config' not in remaining or remaining[remaining.index('--config')+1] != config['native_configuration']:
        raise ValueError('Exact immutable original native configuration required')
    if '--probe-only' in remaining:
        print(json.dumps(dict(required_original_jobs=5, required_epochs_per_job=500,
            original_verified_prior_epochs=238, remaining_full_epochs=262,
            corrected_execution_id=native['recovery_job']['id'],
            original_D_and_G_updates_each=642000, original_admission_commit_GiB=40,
            original_training_function_and_RNG_resume_unchanged=True,
            tensorflow_imported='tensorflow' in sys.modules, numpy_imported='numpy' in sys.modules,
            all_original_experiments_complete=False)))
        return
    namespace = dict(original.__dict__, _retry_admission_settings=admission_settings,
        _dispatch_configuration=config, _dispatch_configuration_path=args.runtime_config)
    path = Path(original.__file__)
    exec(compile(dispatch_ast(path.read_text(encoding='utf-8')), str(path), 'exec'), namespace)
    sys.argv = [str(path)] + remaining
    namespace['main']()


if __name__ == '__main__': main()
