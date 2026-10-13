"""Original full20 SMAP1103 retry after the preserved six-epoch commit abort.

CrossAD author/source/data citations remain in the frozen full139-case queue.
The actual checkpoint saves model.state_dict only; optimizer/RNG continuation
is unavailable, so previous epochs receive zero credit in this fresh attempt.
"""
import ast
from pathlib import Path
from types import FunctionType

from scripts.flow_matching import crossad_gecco1103_full_retry_v1 as previous
from scripts.flow_matching import run_crossad_gecco1103_full_retry_v1 as dispatch

ROOT = previous.ROOT
CANONICAL = 'crossad_author__fresh_train__smap__seed1103'
STATE = 'experiments/runs/fm_crossad_smap1103_full20_retry_v1'
QUEUE = 'configs/experiments/fm_crossad_smap1103_full20_retry_queue.v1.json'
EVIDENCE = 'projects/flow_matching_research/results/2026-10-14/crossad_smap1103_full20_retry_v1'


def build_queue(original):
    return FunctionType(previous.build_queue.__code__,
        dict(previous.build_queue.__globals__, CANONICAL=CANONICAL, STATE_ROOT=STATE))(original)


def weight_only_save_contract(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    saves = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
             and n.func.value.id == 'torch' and n.func.attr == 'save']
    expected = ast.parse('model.state_dict()', mode='eval').body
    if len(saves) != 1 or len(saves[0].args) != 2 or ast.dump(saves[0].args[0]) != ast.dump(expected):
        raise ValueError('Actual author save must contain only model state_dict')
    return True


def verify_incident(root, incident, check_live=True):
    for pin in incident['source_receipts'] + incident['original_partial_artifacts']:
        if previous.digest(root / pin['path']) != pin['sha256']:
            raise ValueError('Preserved SMAP incident/source changed: ' + pin['path'])
    failure = previous.read(root / incident['terminal_failure_path'])
    trace = previous.read(root / incident['training_trace_path'])
    if (failure['exit_code'] != 1 or failure['resources']['resource_guard_aborted'] is not True
            or failure['resources']['abort_resource_snapshot']['commit_available_bytes'] >= 8 * 1024**3
            or trace['epochs_finished'] != 6 or trace['declared_max_epochs'] != 20
            or trace['training_forward_calls'] != 5064 or len(trace['validation_losses']) != 6):
        raise ValueError('Actual six complete original epochs and terminal commit abort required')
    output = root / incident['original_output_directory']
    if (output / 'result.json').exists() or (output / 'stages').exists():
        raise ValueError('Completed native experiment cannot be retrained')
    if (incident['optimizer_rng_resume_available'] is not False
            or incident['completed_budget_credit_for_fresh_retry'] != 0):
        raise ValueError('Weight-only partial training receives zero fresh budget credit')
    weight_only_save_contract(root / incident['weight_save_source'])
    archive = Path(incident['checkpoint_archive']['path'])
    original_checkpoint = output / 'workspace/configs/SMAP/checkpoints_0/checkpoint.pth'
    if (previous.digest(archive) != incident['checkpoint_archive']['sha256']
            or previous.digest(original_checkpoint) != previous.digest(archive)):
        raise ValueError('Keep the actual original weight checkpoint and exact archive')
    if check_live:
        import psutil
        old = incident['stopped_scientific_identity']
        for p in psutil.process_iter(['name', 'create_time', 'cmdline']):
            command = p.info['cmdline'] or []
            if ('python' in (p.info['name'] or '').lower() and '--job-id' in command
                    and (CANONICAL in command or (p.info['create_time'] == old['create_time']
                        and command == old['command']))):
                raise ValueError('Never duplicate a live canonical SMAP scientific worker')
    return trace


def verify(config, root=ROOT, check_live=True):
    namespace = dict(previous.verify.__globals__, CANONICAL=CANONICAL,
        NATIVE_QUEUE=QUEUE, STATE_ROOT=STATE, EVIDENCE=EVIDENCE,
        build_queue=build_queue, verify_incident=verify_incident)
    # Retain all139 original science/data pins, complete native preflight,
    # unchanged dispatch interpreter, recipe and emergency guards.
    return FunctionType(previous.verify.__code__, namespace)(config, root, check_live)


def main():
    FunctionType(dispatch.main.__code__, dict(dispatch.main.__globals__,
        CANONICAL=CANONICAL, verify=verify))()


if __name__ == '__main__':
    main()
