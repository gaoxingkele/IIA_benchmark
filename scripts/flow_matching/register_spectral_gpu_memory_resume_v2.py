"""Register full GPU dispatchers without changing any original scientific job."""
import json
from pathlib import Path

from scripts.flow_matching.run_light_controller import ROOT, digest
from scripts.flow_matching.run_spectral_table2_method_gated_queue_v1 import method_ready


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    original_runtime = 'configs/runtime/fm_spectral_table2_method_gated.v1.json'
    table2 = read(ROOT / original_runtime)
    queue2 = read(ROOT / table2['queue'])
    if len(queue2['jobs']) != 30 or not method_ready(queue2, 'sdformer_ar', table2):
        raise ValueError('Full native six-model/three-dataset/eight-worker SDFormer proof required')
    sources = {r['path'] for r in table2['source_receipts']}
    sources.update([original_runtime, table2['queue'], table2['cpu_preflight'], table2['method_proofs']['sdformer_ar'],
        'scripts/flow_matching/full_gpu_memory_slot_v2.py',
        'scripts/flow_matching/run_spectral_table2_gpu_memory_queue_v2.py',
        'scripts/flow_matching/run_spectral_table3_gpu_memory_queue_v3.py',
        'scripts/flow_matching/register_spectral_gpu_memory_resume_v2.py',
        'tests/test_fm_full_gpu_memory_slot_v2.py'])
    table2 = dict(table2, original_runtime=original_runtime, original_runtime_sha256=digest(ROOT / original_runtime),
        full_job_count=30, admission_policy='Original GPU mutex and unchanged32GiB physical/42GiB commit/14GiB VRAM gates; shared CPU-training mutex is not a GPU reservation. Original guarded emergency16GiB commit/4GiB VRAM protections retained.',
        source_receipts=[dict(path=p, sha256=digest(ROOT / p)) for p in sorted(sources)])
    write_new('configs/runtime/fm_spectral_table2_gpu_memory.v2.json', table2)
    path3 = 'configs/experiments/fm_spectral_table3_original_queue.v2.json'
    queue3 = read(ROOT / path3)
    if len(queue3['jobs']) != 4:
        raise ValueError('Four full long-series original slots required')
    sources3 = {r['path'] for r in queue3['source_receipts']}
    sources3.update([path3, 'scripts/flow_matching/full_gpu_memory_slot_v2.py',
        'scripts/flow_matching/run_spectral_table3_gpu_memory_queue_v3.py',
        'scripts/flow_matching/register_spectral_gpu_memory_resume_v2.py',
        'tests/test_fm_full_gpu_memory_slot_v2.py'])
    registration3 = dict(queue=path3, queue_sha256=digest(ROOT / path3), full_job_count=4,
        settings=queue3['settings'], source_receipts=[dict(path=p, sha256=digest(ROOT / p)) for p in sorted(sources3)],
        admission_policy=table2['admission_policy'], scientific_job_dictionaries_unchanged=True,
        boundary='All full case-specific capacity/evaluator checks remain required before full1000-epoch native training. No preflight promoted to performance; no epoch, batch, evaluator, seed or original-source modification.')
    write_new('configs/runtime/fm_spectral_table3_gpu_memory.v3.json', registration3)
    print(json.dumps(dict(table2_full_jobs=30, sdformer_full_method_proof_passed=True,
        table3_full_jobs=4, original_memory_gates_unchanged=True, source_bindings=len(sources)+len(sources3))))


if __name__ == '__main__':
    main()
