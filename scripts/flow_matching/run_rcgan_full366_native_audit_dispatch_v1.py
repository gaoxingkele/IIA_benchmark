"""Serial original-budget CPU gates, then dispatch the full134-epoch continuation.

No existing training process is stopped or edited. A failed native gate stops
this attempt; it is never repeatedly retried or replaced by a smaller test.
"""
import argparse
from datetime import datetime, timezone
import os
import subprocess

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha
from scripts.flow_matching.rcgan_segmented_checkpoint_evidence_v1 import verify_pins
from scripts.flow_matching.rcgan_seed1104_resume_binding_v3 import cached_resource_api


def verify_dispatch_registration(config):
    verify_pins(config['source_receipts'])
    required = [dict(name='random_order_chain', module='scripts.flow_matching.audit_rcgan_random_order_chain_v3',
                    proof_name='random_order_chain_audit.json', receipt_key='random_order_chain_passed'),
                dict(name='checkpoint_state', module='scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v4',
                    proof_name='checkpoint_resume_state_audit.json', receipt_key='checkpoint_resume_state_passed')]
    if [{k: stage[k] for k in required[0]} for stage in config['stages']] != required:
        raise ValueError('Both complete native gates must run in source order')
    chain = read(ROOT / config['stages'][0]['configuration'])
    state = read(ROOT / config['stages'][1]['configuration'])
    from scripts.flow_matching.rcgan_segmented_checkpoint_evidence_v1 import verify_registration
    queue, job, _, _ = verify_registration(chain)
    resume = read(ROOT / config['resume_configuration'])
    if (state['random_chain_configuration'] != config['stages'][0]['configuration']
            or state['settings'] != chain['settings'] or state['resource_lock'] != chain['resource_lock']
            or resume['checkpoint_state_configuration'] != config['stages'][1]['configuration']
            or resume['canonical_job'] != job or resume['recovery_job']['seed'] != job['seed']
            or resume['verified_completed_epochs'] != 366 or resume['remaining_full_epochs'] != 134
            or resume['full_original_epochs'] != 500 or resume['settings'] != queue['settings']
            or resume['resource_lock'] != queue['resource_lock']
            or any(item['original_scope'] != config['original_scope'] for item in (chain, state, resume))):
        raise ValueError('Preserve every original full native budget, family guard and research scope')


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(ROOT / args.config)
    verify_dispatch_registration(config)
    import psutil
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = cached_resource_api()
    base = ROOT / config['state_root']
    base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('A complete366-state gate dispatcher is already live')
    actor = psutil.Process()
    state = dict(pid=actor.pid, create_time=actor.create_time(), command=actor.cmdline(),
        active_pid=None, required_original_epochs=500, saved_original_epochs=366,
        remaining_original_epochs=134, additional_training_seed_slots=0,
        original_scope=config['original_scope'], benchmark_performance=False,
        all_original_experiments_complete=False)
    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat())
        write(base / 'status.json', state)
    try:
        for stage in config['stages']:
            verify_pins(config['source_receipts'])
            stage_config = read(ROOT / stage['configuration'])
            if stage['name'] == 'random_order_chain':
                from scripts.flow_matching.audit_rcgan_random_order_chain_v3 import bind
            elif stage['name'] == 'checkpoint_state':
                from scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v4 import bind
            else:
                raise ValueError('Both original complete native gates are required')
            verifier = bind()['verify_registration']
            verifier(stage_config)
            stage_base = ROOT / stage_config['state_root']
            stage_base.mkdir(parents=True, exist_ok=True)
            slot_owner = api['exclusive_lock'](stage_base / 'controller.lock')
            if slot_owner is None:
                raise RuntimeError('Preserve an existing independently owned native audit')
            try:
                if (ROOT / stage_config['output_directory']).exists():
                    raise FileExistsError('Preserve every failed or completed native gate')
                def stage_update(values):
                    update(dict(values, stage=stage['name']))
                    write(stage_base / 'status.json', state)
                stage_update(dict(state='waiting_original_native_audit_admission', active_pid=None))
                with api['resource_slot'](ROOT / stage_config['resource_lock'], stage_config['settings'], stage_update):
                    verifier(stage_config)
                    command = [stage_config['python'], '-X', 'utf8', '-u', '-m', stage['module'],
                               '--config', stage['configuration'], '--worker']
                    with (stage_base / 'worker.stdout.log').open('xb') as out, (stage_base / 'worker.stderr.log').open('xb') as err:
                        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                            env=dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'),
                                PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
                                OMP_NUM_THREADS=str(stage_config['cpu_threads']), MKL_NUM_THREADS=str(stage_config['cpu_threads'])),
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                        identity = psutil.Process(child.pid)
                        stage_update(dict(state='executing_complete_original_native_gate', active_pid=child.pid,
                            active_create_time=identity.create_time(), actual_command=identity.cmdline()))
                        code, usage = capped_wait(child, stage_config, api, stage_update)
                proof = ROOT / stage_config['output_directory'] / stage['proof_name']
                passed = code == 0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
                write(stage_base / 'receipt.json', dict(exit_code=code, **usage, **{stage['receipt_key']: passed}))
                stage_update(dict(state='completed_native_gate' if passed else 'failed_native_gate_preserved',
                                  active_pid=None, native_gate_passed=passed))
                if not passed:
                    return
            finally:
                slot_owner.close()
        from scripts.flow_matching.run_rcgan_native_state_recovery_v3 import bind as resume_bind
        resume_bind()['verify_registration'](read(ROOT / config['resume_configuration']))
        command = [config['controller_python'], '-X', 'utf8', '-u', '-m',
                   'scripts.flow_matching.run_rcgan_native_state_recovery_v3',
                   '--config', config['resume_configuration']]
        with (base / 'resume_controller.stdout.log').open('xb') as out, (base / 'resume_controller.stderr.log').open('xb') as err:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                env=dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'),
                    PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1'),
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            identity = psutil.Process(child.pid)
            update(dict(state='both_native_gates_passed_full134_continuation_controller_dispatched',
                resume_controller_pid=child.pid, resume_controller_create_time=identity.create_time(),
                resume_controller_command=identity.cmdline(), active_pid=None,
                full_original500epoch_result_verified=False))
    except Exception as error:
        update(dict(state='dispatcher_failure_preserved', exception_type=type(error).__name__,
                    exception_message=str(error), active_pid=None))
        raise
    finally:
        owner.close()


if __name__ == '__main__':
    main()
