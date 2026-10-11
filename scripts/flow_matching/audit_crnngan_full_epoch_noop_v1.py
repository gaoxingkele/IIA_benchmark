"""Check a complete native source epoch against the preserved native checkpoint."""
import argparse
from pathlib import Path

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha


class CompleteEpochAuditStop(Exception):
    pass


def worker(config):
    import tensorflow as tf
    from scripts.flow_matching import crnngan_native_training_runtime_v1 as original
    from scripts.flow_matching.crnngan_bounded_noop_runtime_v1 import instrumented_main
    for item in config['source_receipts']:
        if sha(ROOT/item['path']) != item['sha256']:
            raise ValueError('Frozen native comparison input changed')
    queue = read(ROOT/config['queue'])
    job = dict(queue['jobs'][0], output_directory=config['native_attempt_directory'])
    reference = read(ROOT/config['reference_progress'])
    expected = reference['epochs'][0]
    atomic_write = original.write
    observations = {}
    session_type = tf.Session

    class ObservedSession(session_type):
        def run(self, fetches, *args, **kwargs):
            value = session_type.run(self, fetches, *args, **kwargs)
            observations['native_graph_operations'] = len(self.graph.get_operations())
            observations['native_noop_operations'] = sum(op.type == 'NoOp' for op in self.graph.get_operations())
            observations['cached_inert_fetch_count'] = sum(op.name == 'iia_native_cached_noop' for op in self.graph.get_operations())
            return value

    def audit_write(path, value):
        atomic_write(path, value)
        if Path(path).name == 'progress.json':
            if value['epochs_completed'] != 1 or value['epochs'][0]['full_training_batches'] != expected['full_training_batches']:
                raise ValueError('Exactly one whole original source epoch/130 full minibatches required for equivalence audit')
            observations['complete_native_progress'] = value
            raise CompleteEpochAuditStop()

    original.instrumented_main = instrumented_main
    original.write = audit_write
    tf.Session = ObservedSession
    try:
        original.run(queue, job)
    except CompleteEpochAuditStop:
        pass
    finally:
        original.write = atomic_write
        tf.Session = session_type
    actual = observations['complete_native_progress']['epochs'][0]
    old_audit = read(ROOT/config['reference_variable_audit'])
    new_audit = read(Path(job['output_directory'])/'native_checkpoints/epoch_000/full_checkpoint_variable_audit.json')
    if (old_audit['tensors'] != new_audit['tensors'] or actual['optimizer_updates'] != expected['optimizer_updates']
            or actual['losses'] != expected['losses']):
        raise ValueError('Complete original first epoch variables/losses/update counts differ')
    from scripts.flow_matching.audit_crnngan_checkpoint_variables_v1 import verify_tensors
    parameters = verify_tensors(tf.train.NewCheckpointReader(str(Path(job['output_directory'])/'native_checkpoints/epoch_000/model')), new_audit)
    if parameters != 4924855 or observations['cached_inert_fetch_count'] != 1:
        raise ValueError('Complete original paper network and single inert fetch required')
    meta = tf.MetaGraphDef()
    meta.ParseFromString(Path(old_audit['meta_graph']).read_bytes())
    observations['reference_native_graph_operations'] = len(meta.graph_def.node)
    observations['reference_native_noop_operations'] = sum(n.op == 'NoOp' for n in meta.graph_def.node)
    write(ROOT/config['output_directory']/'full_epoch_noop_equivalence.json', dict(
        config_sha256=sha(ROOT/config['configuration_path']), full_epoch_native_variables_bitwise_identical=True,
        full_epoch_native_losses_and_updates_identical=True, native_checkpoint_reader_all_tensors_verified=True,
        complete_original_loader_verified=True, full_trainable_parameters=parameters,
        observations=observations, later_curriculum_rng_equivalence_certified=False,
        full500_training_completed=False, benchmark_performance=False, all_original_experiments_complete=False))


def main():
    import subprocess
    import sys
    import psutil
    from datetime import datetime, timezone
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    config = read(ROOT/args.config)
    if args.worker:
        worker(config)
        return
    api = resource_api()
    output = ROOT/config['output_directory']
    output.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](output/'controller.lock')
    if owner is None:
        raise RuntimeError('Native equivalence audit already live')
    state = dict(pid=psutil.Process().pid, create_time=psutil.Process().create_time(), active_pid=None)
    def update(values):
        state.update(values)
        state['captured_utc'] = datetime.now(timezone.utc).isoformat()
        write(output/'status.json', state)
    try:
        with api['resource_slot'](ROOT/config['resource_lock'], config['settings'], update):
            if Path(config['native_attempt_directory']).exists() or (output/'guard_receipt.json').exists():
                raise FileExistsError('Preserve every previous native comparison attempt')
            command = [config['python'], '-X', 'utf8', '-u', '-c',
                "import sys,runpy;sys.path.insert(0,"+repr(str(ROOT))+");sys.argv=['audit','--config',"+repr(args.config)+",'--worker'];runpy.run_module('scripts.flow_matching.audit_crnngan_full_epoch_noop_v1',run_name='__main__')"]
            with (output/'native.stdout.log').open('xb') as out, (output/'native.stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                update(dict(state='running_complete_native_epoch_equivalence', active_pid=child.pid,
                    active_create_time=psutil.Process(child.pid).create_time(), actual_command=command))
                code, usage = capped_wait(child, config, api, lambda values: update(dict(values, state='running_complete_native_epoch_equivalence')))
            passed = code == 0 and (output/'full_epoch_noop_equivalence.json').exists()
            write(output/'guard_receipt.json', dict(exit_code=code, passed=passed, **usage))
            update(dict(state='completed' if passed else 'failed_preserved', active_pid=None))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
