"""Preserve failed capacity evidence and register all full original method checks."""
import json
from pathlib import Path

from scripts.flow_matching.run_light_controller import ROOT,digest
from scripts.flow_matching.run_spectral_table2_original_v1 import fingerprint,verify


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path,value):
    destination=ROOT/path
    if destination.exists():raise FileExistsError('Registration already frozen')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def receipts(paths):
    return [dict(path=p,sha256=digest(ROOT/p)) for p in dict.fromkeys(paths)]


def main():
    queue_path='configs/experiments/fm_spectral_table2_original_queue.v1.json'
    queue=read(ROOT/queue_path);verify(queue)
    previous=read(ROOT/'configs/runtime/fm_spectral_table2_gpu_memory.v2.json')
    failed_root='experiments/runs/fm_spectral_table2_gpu_retry_v2'
    failure=read(ROOT/failed_root/'resource_receipt.json')
    if failure['full_diagnostic_passed'] or not failure['resource_guard_aborted']:
        raise ValueError('Expected incomplete full capacity attempt is absent')
    proof='experiments/runs/fm_spectral_table2_original_v1/preflight/gpu_spectral_v3.json'
    if (ROOT/proof).exists():raise FileExistsError('Full Spectral proof already exists')
    settings=dict(previous['settings']);settings.pop('controller_yield_seconds')
    paths=[queue_path,'configs/runtime/fm_spectral_table2_gpu_memory.v2.json',
        failed_root+'/status.json',failed_root+'/resource_receipt.json',failed_root+'/stdout.log',failed_root+'/stderr.log',
        'scripts/flow_matching/run_spectral_table2_spectral_preflight_v3.py',
        'scripts/flow_matching/run_registered_full_gpu_command_v2.py',
        'scripts/flow_matching/register_spectral_capacity_resume_v3.py',
        'tests/test_fm_spectral_capacity_preflight_v3.py',
        'scripts/flow_matching/run_spectral_table2_original_v1.py',
        'scripts/flow_matching/register_spectral_table2_original_v1.py',
        'scripts/flow_matching/full_gpu_memory_slot_v2.py',
        'scripts/flow_matching/run_spectral_original_queue.py',
        'scripts/flow_matching/run_grasp_protocol_queue.py',
        'scripts/flow_matching/run_tsad_queue.py','scripts/flow_matching/heavy_resources.py',
        'scripts/flow_matching/heavy_scheduler_v2.py',
        'tests/test_fm_full_gpu_memory_slot_v2.py','tests/test_fm_table2_method_admission_v1.py']
    runtime=dict(queue=queue_path,command=[str(ROOT/queue['python']),'-X','utf8','-u','-m',
        'scripts.flow_matching.run_spectral_table2_spectral_preflight_v3','--queue',queue_path,'--output',proof],
        expected_output=proof,expected_queue_fingerprint=fingerprint(queue),
        state_root='experiments/runs/fm_spectral_table2_spectral_preflight_v3',
        resource_lock=previous['resource_lock'],settings=settings,source_receipts=receipts(paths),
        boundary='Three complete Spectral model/full-batch backward/optimizer checks and all three original eight-worker data loaders. Same original architecture, native source operators and data shapes. Preserve terminal mixed-method preflight failure. Full32GiB RAM/42GiBcommit/14GiBVRAM admission,16GiBcommit/4GiBVRAM emergency guards; previous diagnostic30GiBcommit gate was insufficient. No benchmark score from diagnostic.')
    preflight_path='configs/runtime/fm_spectral_table2_spectral_preflight.v3.json'
    write_new(preflight_path,runtime)
    resumed=dict(previous,method_proofs=dict(previous['method_proofs'],spectral_flow=proof))
    resumed['source_receipts']=receipts([r['path'] for r in previous['source_receipts']]+paths+[preflight_path])
    resumed['boundary']='All30 original Table2 full training/sampling/evaluation jobs remain unchanged. Only full Spectral proof path points to the newly registered method-specific capacity diagnostic; keep existing complete SDFormer proof, all full GPU resource settings and all scientific/evaluator budgets. Switch only a verified idle waiting controller.'
    queue_runtime='configs/runtime/fm_spectral_table2_gpu_memory.v3.json'
    write_new(queue_runtime,resumed)
    print(json.dumps(dict(preflight_runtime=preflight_path,full_jobs=len(queue['jobs']),resumed_runtime=queue_runtime,
        previous_failure_preserved=True,original_scientific_queue_sha256=digest(ROOT/queue_path),
        resource_settings=settings)))


if __name__=='__main__':main()
