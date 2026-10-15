"""Recover proven pre-inference metadata rejections without changing MOMENT math.

The official checkpoint config seq_len is 512 for both registered contexts.
The external TAB100 window remains 100 and uses the original extra-patch policy.
Only the source validator's mistaken equality between those two is replaced.
An isolated module retains all original inference, scores and independent math.
Every admitted case needs a terminal v1 rejection before any native artifact.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

from scripts.flow_matching import run_moment_full_checkpoint_replay_v1 as original

ROOT, read, write, sha = original.ROOT, original.read, original.write, original.sha
MODULE = 'scripts.flow_matching.run_moment_TAB100_metadata_recovery_v2'
CONFIG = 'configs/runtime/fm_moment_TAB100_metadata_recovery.v2.json'
ORIGINAL = 'configs/runtime/fm_moment_full78_checkpoint_replay.v1.json'
SOURCE = 'scripts/flow_matching/run_moment_full_checkpoint_replay_v1.py'
TEST = 'tests/test_fm_moment_TAB100_metadata_recovery_v2.py'
OLD_COMPARISON = "audit['configured_sequence_length'] != job['window']"
NEW_COMPARISON = "audit['configured_sequence_length'] != _registered_checkpoint_length(case)"


def registered_checkpoint_length(case):
    parameters = case['config']['parameters']
    path = ROOT/parameters['checkpoint_directory']/'config.json'
    if sha(path) != parameters['config_sha256']:
        raise ValueError('Official checkpoint configuration hash changed')
    length = read(path)['seq_len']
    if type(length) is not int or length != 512:
        raise ValueError('Require the exact official registered sequence length 512')
    return length


def corrected_private_module(source):
    """Exactly one Compare AST changes; the numerical module body is identical."""
    tree = ast.parse(source)
    old = ast.dump(ast.parse(OLD_COMPARISON, mode='eval').body)
    changes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'check_source':
            class Correction(ast.NodeTransformer):
                def visit_Compare(self, candidate):
                    if ast.dump(candidate) == old:
                        changes.append(ast.dump(candidate))
                        return ast.copy_location(ast.parse(NEW_COMPARISON, mode='eval').body, candidate)
                    return self.generic_visit(candidate)
            Correction().visit(node)
    if len(changes) != 1:
        raise ValueError('Require exactly the diagnosed original metadata predicate')
    ast.fix_missing_locations(tree)
    private = types.ModuleType('_moment_TAB100_corrected_private_v2')
    private.__file__ = str(ROOT/SOURCE)
    exec(compile(tree, str(ROOT/SOURCE), 'exec'), private.__dict__)
    private._registered_checkpoint_length = registered_checkpoint_length
    private.MODULE = MODULE
    return private


def registration():
    parent = read(ROOT/ORIGINAL)
    cases = original.verify(parent)
    return dict(schema_version=1, original_runtime=ORIGINAL, original_runtime_sha256=sha(ROOT/ORIGINAL),
        required_original_cases=78, original_scope=parent['original_scope'],
        selected_ids=[i for i, c in cases.items() if c['job']['window'] == 100],
        settings=parent['settings'], resource_lock=parent['resource_lock'],
        state_root='experiments/runs/fm_moment_TAB100_metadata_recovery_v2',
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_TAB100_metadata_recovery_v2',
        controller_python=parent['controller_python'],
        source_receipts=[dict(path=p, sha256=sha(ROOT/p)) for p in [SOURCE, __file_relative(), TEST]],
        correction=dict(old_comparison=OLD_COMPARISON, new_comparison=NEW_COMPARISON,
            official_checkpoint_sequence_length=512, external_window=100,
            native_and_independent_numerical_bodies_changed=False),
        citation='https://arxiv.org/abs/2402.03885',
        data_citation='https://github.com/decisionintelligence/TAB',
        boundary='All39 registered context100 slots remain separate from context512 and original78 scope. '
            'Only proven v1 pre-native metadata rejections may enter new isolated attempts. '
            'No numerical training/inference retry,changed windows,padding,weights,data,thresholds,tolerances or resource policy. '
            'Historical guard gaps,preprocessing reconstruction,TAB harness and paper equivalence remain unproven.')


def __file_relative():
    return Path(__file__).resolve().relative_to(ROOT).as_posix()


def verify(config):
    if config != registration():
        raise ValueError('Unchanged correction,all39 contexts,source pins and original resource policy required')
    parent = read(ROOT/ORIGINAL)
    cases = original.verify(parent)
    parent = dict(parent, state_root=config['state_root'], artifact_root=config['artifact_root'])
    return parent, {i: cases[i] for i in config['selected_ids']}


def metadata_failure(case, parent):
    """An exit code alone never permits another model attempt."""
    identity = case['job']['id']
    prior = read(ROOT/ORIGINAL)
    attempt = ROOT/prior['state_root']/'attempts'/identity/'native'
    receipt = attempt/'guard_receipt.json'
    if not receipt.exists():
        return None
    guard = read(receipt)
    expected = [str(ROOT/case['queue']['python']), '-B', '-X', 'utf8', '-u', '-m', original.MODULE,
                '--config', ORIGINAL, '--stage', 'native', '--job-id', identity]
    if guard['passed'] or guard['exit_code'] != 1 or guard['resource_guard_aborted'] or guard['proof_sha256'] is not None:
        raise ValueError('Not the terminal pre-native metadata rejection')
    if guard['exact_child_handle']['command'] != expected:
        raise ValueError('Original failed child identity differs')
    import psutil
    handle = guard['exact_child_handle']
    try:
        live = psutil.Process(handle['pid'])
        if abs(live.create_time()-handle['create_time']) < .01:
            raise ValueError('Original child still live; no recovery admission')
    except psutil.NoSuchProcess:
        pass
    stderr = (attempt/'stderr.log').read_text(encoding='utf-8')
    if ('result,dataset,binding = check_source(runtime,case)' not in stderr
            or "raise ValueError('Full unchanged official-weight zero-shot result required')" not in stderr
            or not stderr.rstrip().endswith('ValueError: Full unchanged official-weight zero-shot result required')
            or (Path(prior['artifact_root'])/identity/'native').exists()):
        raise ValueError('Failure location must precede model inference and native artifact creation')
    private = corrected_private_module((ROOT/SOURCE).read_text(encoding='utf-8'))
    # Demonstrate the exact old false predicate and the exact pinned config value.
    source = read(ROOT/case['job']['output_directory']/'result.json')
    audit = source['checkpoint_load_audit']
    if case['job']['window'] != 100 or audit['configured_sequence_length'] != registered_checkpoint_length(case):
        raise ValueError('Not the diagnosed 512-versus100 metadata defect')
    private.check_source(parent, case)  # All other original source checks must pass.
    return dict(original_guard_path=str(receipt), original_guard_sha256=sha(receipt),
                original_stderr_sha256=sha(attempt/'stderr.log'), diagnosed_pre_inference_only=True,
                official_checkpoint_seq_len=512, unchanged_external_window=100)


def controller(config, runtime, cases, private):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = install_facades()
    directory = ROOT/config['state_root']; directory.mkdir(parents=True, exist_ok=True)
    lane = api['exclusive_lock'](directory/'controller.lock')
    if lane is None:
        raise RuntimeError('Metadata recovery owner already live')
    verified, failed, admitted = [], {}, {}
    common = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), required_cases=len(cases),
                  original_scope_required_cases=78, original_scope=config['original_scope'],
                  all_original_experiments_complete=False)
    state = dict(common)
    def update(value=None, **kw):
        state.update(value or {}); state.update(kw)
        state.update(verified_ids=list(verified), verified_count=len(verified), failed_preserved=dict(failed),
                     diagnosed_admissions=dict(admitted), captured_utc=datetime.now(timezone.utc).isoformat())
        write(directory/'status.json', state)
    def stage(case, phase):
        identity = case['job']['id']; attempt = directory/'attempts'/identity/phase
        proof = Path(runtime['artifact_root'])/identity/('native/native_replay.json' if phase == 'native' else 'independent/review.json')
        receipt = attempt/'guard_receipt.json'
        if receipt.exists():
            previous = read(receipt)
            if (previous['passed'] and previous['exit_code'] == 0 and not previous['resource_guard_aborted']
                    and sha(proof) == previous['proof_sha256'] and read(proof)['passed']):
                _, _, binding = private.check_source(runtime, case)
                if read(proof)['source_binding'] != binding:
                    raise ValueError('Existing recovered proof binding changed')
                return True
            raise ValueError('Preserve failed recovery; no automatic numeric retry')
        if attempt.exists() or proof.exists() or (phase == 'native' and proof.parent.exists()):
            raise ValueError('Preserve partial recovered attempt for diagnosis')
        attempt.mkdir(parents=True, exist_ok=False)
        command = [str(ROOT/case['queue']['python']), '-B', '-X', 'utf8', '-u', '-m', MODULE,
                   '--config', CONFIG, '--stage', phase, '--job-id', identity]
        with (attempt/'stdout.log').open('xb') as out, (attempt/'stderr.log').open('xb') as err:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1',
                         TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                         PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src')),
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            handle = dict(pid=child.pid, create_time=psutil.Process(child.pid).create_time(), command=command)
            write(attempt/'launch.json', handle)
            update(state='running_corrected_metadata_full_'+phase, active_pid=child.pid, active_create_time=handle['create_time'], job=identity)
            code, usage = capped_wait(child, runtime, api, update)
        passed = code == 0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof).get('passed') is True
        write(receipt, dict(passed=passed, exit_code=code, **usage, proof_sha256=sha(proof) if proof.exists() else None, exact_child_handle=handle))
        return passed
    try:
        while len(verified)+len(failed) < len(cases):
            for identity, case in cases.items():
                if identity in verified or identity in failed:
                    continue
                try:
                    # Cache only the accepted immutable binding; each worker rechecks hashes.
                    if identity not in admitted:
                        evidence = metadata_failure(case, runtime)
                        if evidence is None:
                            continue
                        admitted[identity] = evidence
                        write(directory/'admissions'/(identity+'.json'), evidence)
                    def waiting(value):
                        update(dict(value, active_pid=None, job=identity))
                    with api['resource_slot'](ROOT/runtime['resource_lock'], runtime['settings'], waiting, poll_seconds=10):
                        if not stage(case, 'native') or not stage(case, 'independent'):
                            raise ValueError('Guarded corrected-metadata stage failed; preserve it')
                    verified.append(identity)
                except Exception as error:
                    failed[identity] = str(error)
                update(state='scanning_proven_pre_native_metadata_rejections', active_pid=None, job=None)
            update(state='waiting_proven_pre_native_metadata_rejection_or_original_source', active_pid=None, job=None)
            time.sleep(10)
        update(state='all39_context100_attempts_terminal_other_experiments_remain', active_pid=None, job=None)
    finally:
        lane.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--stage', choices=['native', 'independent'])
    parser.add_argument('--job-id')
    args = parser.parse_args()
    if args.config != CONFIG:
        raise ValueError('Use the exact registered recovery configuration')
    if args.register:
        target = ROOT/CONFIG
        if target.exists():
            raise FileExistsError('Preserve the frozen correction registration')
        config = registration()
        with target.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(config, stream, indent=2); stream.write('\n')
        print(json.dumps(dict(configuration=CONFIG, selected_cases=len(config['selected_ids']), original_cases=78)))
        return
    config = read(ROOT/CONFIG); runtime, cases = verify(config); runtime['_config_path'] = CONFIG
    private = corrected_private_module((ROOT/SOURCE).read_text(encoding='utf-8'))
    if args.probe_only:
        admissions = {identity: evidence for identity, case in cases.items() if (evidence := metadata_failure(case, runtime)) is not None}
        print(json.dumps(dict(passed=True, selected_cases=len(cases), original_required_cases=78, proven_pre_native_rejections=admissions,
                             numerical_workers_started=0, torch_imported='torch' in sys.modules, numpy_imported='numpy' in sys.modules)))
    elif args.stage:
        case = cases[args.job_id]
        if metadata_failure(case, runtime) is None:
            raise ValueError('Actual prior metadata rejection is required')
        (private.native_worker if args.stage == 'native' else private.review_worker)(runtime, case)
    else:
        controller(config, runtime, cases, private)


if __name__ == '__main__':
    main()
