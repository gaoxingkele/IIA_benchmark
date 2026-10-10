"""Preserve the interrupted570/1500-epoch ablation and retry its original full slot."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import psutil

from scripts.flow_matching.run_light_controller import ROOT,digest
from scripts.flow_matching.grasp_protocol_runner import complete,write_json,verify
from scripts.flow_matching.native_exact_recovery_manifest_v2 import verified_manifest


def main():
    original_path='configs/experiments/fm_grasp_protocol_queue.v4.json'
    target=ROOT/'configs/experiments/fm_native_grasp_tau025_exact_recovery.v3.json'
    if target.exists():raise FileExistsError('Exact recovery already registered')
    original=json.loads((ROOT/original_path).read_text());verify(original)
    job=next(j for j in original['jobs'] if j['profile']=='tau_0p25' and j['dataset']=='swan' and j['seed']==1103)
    if complete(job):raise ValueError('Original complete preferred')
    output=ROOT/job['output_directory']
    progress_path=output/'progress.json';history_path=output/'training_history.jsonl'
    progress=json.loads(progress_path.read_text())
    if progress['epoch']!=570 or progress['optimizer_updates']!=35340:
        raise ValueError('Interrupted full native attempt changed')
    states=json.loads((ROOT/'experiments/runs/fm_grasp_protocol_v4/status.json').read_text())
    if states['jobs'][job['id']]!='failed_or_partial_preserved':
        raise ValueError('Current original scheduler still considers this slot active')
    commands=[]
    for process in psutil.process_iter(['pid','name']):
        if (process.info['name'] or '').lower()!='python.exe':continue
        try:
            command=process.cmdline()
            if job['id'] in command:raise ValueError('Corresponding original worker still live')
            commands.append(dict(pid=process.pid,create_time=process.create_time(),command=command))
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    evidence_path='projects/flow_matching_research/results/2026-10-11/grasp_tau025_exact_recovery_v3/terminal_original_attempt.json'
    evidence=ROOT/evidence_path
    if evidence.exists():raise FileExistsError('Terminal interruption evidence already captured')
    write_json(evidence,dict(captured_utc=datetime.now(timezone.utc).isoformat(),original_id=job['id'],
        progress=progress,original_progress_sha256=digest(progress_path),original_history_sha256=digest(history_path),
        no_matching_live_model_process=True,original_queue_status=states['jobs'][job['id']],
        observed_python_processes=commands,full_budget=dict(epochs=1500,optimizer_updates=job['expected_optimizer_updates']),
        boundary='Original stopped at570/1500epochs with35340/93000updates; no full result. Preserve partial files and start exact original full-budget seed in a new directory; do not invent an optimizer/RNG resume checkpoint.'))
    queue=deepcopy(original)
    queue.update(recovery_track='grasp',recovery_original_queue=original_path,
        recovery_original_queue_sha256=digest(ROOT/original_path),recovery_original_job_ids=[job['id']],
        state_root='experiments/runs/fm_native_grasp_tau025_exact_recovery_v3',
        artifact_root='F:/aicoding/IIA_Data/experiments/native_grasp_tau025_exact_recovery_v3',
        lock_poll_seconds=.25,cohort_count=1,
        recovery_policy='One interrupted original SWAN tau0.25 seed1103 slot, exact1500-epoch/93000-update retry. Original files and full scientific settings unchanged; no additional seed or metric selection.',
        recovery_failure_receipts=[dict(original_job_id=job['id'],path=evidence_path,sha256=digest(evidence))])
    recovered=deepcopy(job)
    recovered.update(id=job['id']+'__native_exact_recovery_v3',recovery_original_job_id=job['id'],
        output_directory=queue['state_root']+'/jobs/'+job['id']+'__native_exact_recovery_v3')
    if (ROOT/recovered['output_directory']).exists():raise FileExistsError('Recovery output already exists')
    queue['jobs']=[recovered]
    paths=[original_path,evidence_path,progress_path.relative_to(ROOT).as_posix(),history_path.relative_to(ROOT).as_posix(),
        'scripts/flow_matching/native_exact_recovery.py','scripts/flow_matching/native_exact_recovery_manifest_v2.py',
        'scripts/flow_matching/run_native_grasp_recovery_v2.py',
        'scripts/flow_matching/run_native_grasp_gpu_memory_recovery_v3.py',
        'scripts/flow_matching/run_grasp_gpu_memory_fair_queue_v6.py',
        'scripts/flow_matching/full_gpu_memory_slot_v2.py',
        'scripts/flow_matching/register_grasp_tau025_exact_recovery_v3.py',
        'tests/test_fm_native_grasp_gpu_recovery_v3.py']
    queue['source_receipts']+=[dict(path=p,sha256=digest(ROOT/p)) for p in paths]
    verified_manifest(queue)
    write_json(target,queue)
    print(json.dumps(dict(original_slot=job['id'],full_epochs=1500,full_optimizer_updates=job['expected_optimizer_updates'],
                         partial_original_preserved=True,queue=target.relative_to(ROOT).as_posix())))


if __name__=='__main__':main()
