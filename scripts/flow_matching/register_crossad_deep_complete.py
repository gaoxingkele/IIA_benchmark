"""Freeze a new unlaunched branch; never rewrite a prior memory-protected attempt."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import sha,write_json


def main():
    path=ROOT/'configs/experiments/fm_crossad_complete_queue.v3.json';queue=json.loads(path.read_text())
    queue['schema_version']=4;settings=queue['settings'];settings.update(queue_path='configs/experiments/fm_crossad_complete_queue.v4.json',
        preflight_report='docs/reports/fm_crossad_deep_preflight_2026-10-09.json')
    common=['src/iia_benchmark/models/fm_crossad_memory_v2.py','scripts/flow_matching/run_crossad_memory_v2_job.py',
        'scripts/flow_matching/run_crossad_ablation_v2_job.py','scripts/flow_matching/register_crossad_deep_complete.py']
    for job in queue['jobs']:
        assert not (ROOT/job['output_directory']/'workspace').exists()
        job['runner']='scripts/flow_matching/run_crossad_ablation_v2_job.py' if job['reconstructed_ablation'] else 'scripts/flow_matching/run_crossad_memory_v2_job.py'
        job['activation_checkpointing']='pure_layer_v2'
        job['frozen_files'] += [{'path':p,'sha256':sha(ROOT/p)} for p in common]
    queue['superseded_unlaunched_queues'].append(path.relative_to(ROOT).as_posix())
    destination=ROOT/settings['queue_path']
    if destination.exists() and json.loads(destination.read_text())!=queue:raise ValueError('Deep queue frozen')
    write_json(destination,queue)
    write_json(ROOT/'docs/reports/fm_crossad_deep_registration_2026-10-09.json',{'registered_jobs':139,'queue_path':settings['queue_path'],
        'queue_sha256':sha(destination),'unique_frozen_files':len({f['path'] for j in queue['jobs'] for f in j['frozen_files']}),
        'boundary':'Full original batches/budgets retained; pure-layer replay plus unused diagnostic attention reference release. No source production runs from superseded prototypes. All full-native-batch checks still required.','all_experiments_complete':False})
    print('Registered full 139-job pure-layer checkpoint branch',sha(destination))


if __name__=='__main__':main()
