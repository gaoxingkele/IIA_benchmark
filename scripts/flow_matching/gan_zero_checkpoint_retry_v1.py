"""Preserve zero-checkpoint memory aborts and require stronger full-budget retry.

TimeGAN (Yoon et al., NeurIPS2019) and RCGAN (Esteban et al.) retain the frozen
author source/data citations, algorithms, seeds and complete original queues.
Unsaved work is never credited as completed stages or epochs.
"""
import ast
import copy
from pathlib import Path
import sys

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha
from scripts.flow_matching.rcgan_seed1106_resume_binding_v1 import stronger_admission


def verify_zero_checkpoint_incident(incident, check_live=True):
    for pin in incident['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Preserved actual zero-checkpoint incident changed')
    receipt = read(ROOT / incident['terminal_receipt'])
    if (receipt['exit_code'] != 1 or receipt['resource_guard_aborted'] is not True
            or receipt['full_native_result_verified'] is not False
            or receipt['abort_snapshot']['commit_available_bytes'] >= 16 * 1024 ** 3):
        raise ValueError('Actual terminal original commit-guard abort required')
    output = ROOT / incident['canonical_job']['output_directory']
    if any((output / name).exists() for name in ('result.json', 'progress.json', 'checkpoints', 'native_checkpoints', 'completed_full_evaluators.json')):
        raise ValueError('Any saved original work requires complete state recovery instead')
    if check_live:
        import psutil
        expected = incident['stopped_native_identity']
        for process in psutil.process_iter(['create_time', 'cmdline']):
            if (process.info['create_time'] == expected['create_time']
                    and process.info['cmdline'] == expected['command']):
                raise ValueError('Never retry a live original scientific process')
    elif Path(sys.executable).resolve() != Path(incident['native_python']).resolve():
        raise ValueError('Only the frozen native worker may defer host process monitoring')
    return receipt


def execution_job(config, canonical):
    if canonical['id'] != config['zero_checkpoint_retry']['canonical_job']['id']:
        return canonical
    if canonical != config['zero_checkpoint_retry']['canonical_job']:
        raise ValueError('Retry must retain the exact original canonical identity')
    actual = copy.deepcopy(canonical)
    actual['output_directory'] = config['zero_checkpoint_retry']['execution_output_directory']
    if actual['output_directory'] == canonical['output_directory']:
        raise ValueError('Preserve the original failed directory')
    return actual


def stronger_slot_ast(tree, config_name, expected_argument):
    reference = ast.dump(ast.parse(expected_argument, mode='eval').body)
    changes = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'api'):
            continue
        key = node.func.slice
        if isinstance(key, ast.Index):
            key = key.value
        value = key.value if isinstance(key, ast.Constant) else key.s if isinstance(key, ast.Str) else None
        if value != 'resource_slot':
            continue
        if len(node.args) != 3 or ast.dump(node.args[1]) != reference:
            raise ValueError('Only the original resource admission argument may change')
        node.args[1] = ast.Call(func=ast.Name(id='_stronger_admission', ctx=ast.Load()),
            args=[ast.Name(id=config_name, ctx=ast.Load()), node.args[1]], keywords=[])
        changes += 1
    if changes != 1:
        raise ValueError('Exactly one stronger admission call required')
    return ast.fix_missing_locations(tree)
