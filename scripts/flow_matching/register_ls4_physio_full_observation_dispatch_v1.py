"""Register all original full LS4 patient evaluations without changing native science."""
from datetime import datetime,timezone
from pathlib import Path
import json
import os
import psutil
from scripts.flow_matching.run_ls4_physio_full_observation_dispatch_v1 import ROOT,read,sha,verify,all_jobs_unstarted
from scripts.flow_matching.ls4_physio_full_evaluation_observation_v1 import exclusive


def verified_idle_console_children(process):
    """Permit only the original verified Windows console host, never a worker."""
    expected=os.path.normcase(str(Path(os.environ['SystemRoot'])/'System32'/'conhost.exe'))
    children=[]
    for child in process.children(recursive=True):
        command=child.cmdline();created=child.create_time()
        executable=os.path.normcase(child.exe())
        command_executable=command[0].removeprefix('\\??\\') if command else ''
        if (child.name().lower()!='conhost.exe' or executable!=expected
                or os.path.normcase(command_executable)!=expected or command[1:]!=['0x4']
                or not 0<=created-process.create_time()<1):
            raise ValueError('Unexpected child; do not hand off any native training or unknown process')
        children.append(dict(pid=child.pid,create_time=created,executable=child.exe(),actual_command=command,
            verified_original_windows_console_host=True))
    return children


def main():
    path='configs/runtime/fm_ls4_physio_full_observation_dispatch.v1.json'
    original='configs/experiments/fm_ls4_native_physionet_queue.v3.json'
    queue=read(ROOT/original)
    if not all_jobs_unstarted(queue):raise ValueError('Every original full LS4 attempt must remain unstarted before handoff')
    state_path=queue['state_root']+'/status.json';state=read(ROOT/state_path)
    p=psutil.Process(state['pid'])
    console_children=verified_idle_console_children(p)
    if (p.create_time()!=state['create_time'] or state['active_pid'] is not None
            or state['jobs']!={} or state['state']!='waiting_full_gpu_resources'
            or not any('run_ls4_native_physionet_queue_v3' in value for value in p.cmdline())):
        raise ValueError('Exact idle original native LS4 queue with no training child required')
    root='experiments/runs/fm_ls4_physio_full_observation_dispatch_v1'
    sources={row['path'] for row in queue['source_receipts']}
    sources.update([original,queue['environment_lock'],
        'scripts/flow_matching/ls4_physio_full_evaluation_observation_v1.py',
        'scripts/flow_matching/run_ls4_physio_full_observation_dispatch_v1.py',
        'scripts/flow_matching/register_ls4_physio_full_observation_dispatch_v1.py',
        'tests/test_fm_ls4_physio_full_evaluation_observation_v1.py'])
    sources.update(j['model_config'] for j in queue['jobs'])
    config=dict(runtime_configuration=path,original_queue=original,state_root=root,
        handoff_receipt=root+'/handoff.json',old_owner=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline(),state_path=state_path,
            verified_original_console_children=console_children),
        full_original_jobs=queue['jobs'],original_settings=queue['settings'],original_gpu_mutex=queue['resource_lock'],
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        required_full_native_evaluations=500,required_full_native_reconstruction_and_nll_calls=37500,
        required_original_training_optimizer_updates=750000,additional_training_seed_slots=0,
        citation='Zhou et al. Long Range Probabilistic Forecasting in Multivariate Time Series with Latent State Space Models. ICML2023.',
        author_source='https://github.com/alexzhou907/ls4',
        boundary='Retain all original10 complete patient jobs: interpolation and extrapolation, seeds42/1103..1106, '
            '500 full epochs, original6400/1600 patients and41 variables, native64/32 batches, original Cauchy correction '
            'and all750000 optimizer updates. Preserve every500 full native evaluation model/optimizer/scheduler, '
            'all CPU/CUDA/NumPy/Python/loader-generator RNG states, original train/gradient/backend flags and every '
            'original reconstruction/NLL input and output tensor. Wrap original calls without new random draws, '
            'optimizer updates or output replacement. Original complete worker/controller AST retained except '
            'observation binding/result fields, worker entry/config argument and queue state root. '
            'Native test-MSE checkpoint selection and global normalization remain explicit original protocol limits. '
            'Observation registration/cold probes/fixture tests do not prove trained numerical equivalence, '
            'paper result equality or TAB certification. All full original scope remains unfinished.',
        source_receipts=[dict(path=s,sha256=sha(ROOT/s)) for s in sorted(sources)])
    verify(config);exclusive(ROOT/path,config)
    exclusive(ROOT/'projects/flow_matching_research/results/2026-10-12/ls4_physio_full_observation_v1/registration.json',
        dict(captured_utc=datetime.now(timezone.utc).isoformat(),configuration=path,configuration_sha256=sha(ROOT/path),
            source_pins=len(config['source_receipts']),full_original_jobs=queue['jobs'],old_actual_idle_owner=config['old_owner'],
            required_full_native_evaluations=500,required_full_native_method_calls=37500,
            original_scope=config['original_scope'],native_training_stopped_or_modified=False,
            full_original_training_results_complete=0,new_numeric_results_published=0,all_original_experiments_complete=False))
    print(json.dumps(dict(configuration=path,source_pins=len(config['source_receipts']),original_full_jobs=10)))


if __name__=='__main__':main()
