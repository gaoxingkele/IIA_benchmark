"""Pure two-case full native batch capacity controller with unchanged GPU guards."""
import argparse
from datetime import datetime, timezone
import os
import subprocess
import sys
import uuid
import psutil
from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read, sha, verify_receipts
from scripts.flow_matching.run_flow_mismatching_full_training_queue_v1 import queue_api
from scripts.flow_matching.run_tccm_full_image_evaluation_queue_v1 import full_gpu_slot
from scripts.flow_matching.run_resource_gated_light_controller_v1 import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config',required=True);parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args();runtime=read(ROOT/args.runtime_config);verify_receipts(runtime['source_receipts'])
    config=read(ROOT/runtime['configuration']);verify_receipts(config['source_receipts'])
    if [x['objective'] for x in config['capacity_cases']]!=['WT','FM']:
        raise ValueError('Both full original WT and FM capacity cases required')
    if args.validate_only:
        print(dict(full_native_capacity_cases=2,batch_size=32,model_parameters=45289539,
            torch_imported='torch' in sys.modules,numpy_imported='numpy' in sys.modules,benchmark_performance=False));return
    api=queue_api();lane=api['exclusive_lock'](ROOT/runtime['lane_lock'])
    if lane is None:raise RuntimeError('Existing actual full native CUDA capacity controller owns the lane')
    process=psutil.Process();state=dict(pid=process.pid,create_time=process.create_time(),active_pid=None,complete=False)
    def update(value):
        state.update(value,captured_utc=datetime.now(timezone.utc).isoformat());write(ROOT/runtime['state_root']/'status.json',state)
    try:
        for case in config['capacity_cases']:
            output=ROOT/case['output_directory']
            if (output/'capacity_probe_result.json').exists():
                raise FileExistsError('Preserve any earlier full CUDA capacity attempt; explicit review required')
            update(dict(case_id=case['id'],state='waiting_for_original_memory_and_GPU_guards',active_pid=None))
            with full_gpu_slot(api,runtime,update):
                attempt=output/'attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True,exist_ok=False)
                command=[runtime['python'],'-X','utf8','-u','-m','scripts.flow_matching.probe_full_image_cuda_execution_recovery_v2',
                    '--config',runtime['configuration'],'--case-id',case['id'],'--attempt-directory',str(attempt.resolve())]
                with (attempt/'stdout.log').open('wb') as stdout,(attempt/'stderr.log').open('wb') as stderr:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,
                        env={**os.environ,'PYTHONUTF8':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
                            'CUDA_VISIBLE_DEVICES':runtime['settings']['gpu_uuid'],'CUBLAS_WORKSPACE_CONFIG':':4096:8'})
                    identity=psutil.Process(child.pid)
                    update(dict(state='running',active_pid=child.pid,active_create_time=identity.create_time(),
                        actual_command=command,shared_lock_owned=True))
                    code,usage=api['guarded_gpu_wait'](child,runtime['settings'],api,update)
                terminal=dict(returncode=code,worker_pid=child.pid,worker_create_time=state['active_create_time'],
                    command=command,resource_usage=usage)
                write(attempt/'terminal.json',terminal)
                update(dict(active_pid=None,last_terminal=terminal,shared_lock_owned=False))
                if code!=0 or usage['resource_guard_aborted']:
                    update(dict(state='failed_capacity_attempt_preserved',complete=False));return
                location=read(output/'capacity_probe_result.json');verify_receipts([location]);report=read(location['path'])
                if (not report['full_batch_forward_backward_optimizer_and_checkpoint_IO_passed']
                        or report['actual_batch_shape']!=[32,3,128,128] or report['full_model_parameters']!=45289539
                        or report['actual_optimizer_updates']!=1 or report['benchmark_performance']):
                    raise ValueError('Actual complete native batch32 capacity and full checkpoint IO required')
        update(dict(state='both_full_native_batch32_cuda_capacity_cases_passed',complete=True,active_pid=None,
            benchmark_performance=False,all_original_experiments_complete=False))
    finally:lane.close()


if __name__=='__main__':main()
