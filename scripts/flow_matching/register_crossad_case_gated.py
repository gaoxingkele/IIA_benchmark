"""Keep every job; let validated native cases execute while other cases remain pending."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import sha,write_json


def main():
    parent=ROOT/'configs/experiments/fm_crossad_complete_queue.v4.json';queue=json.loads(parent.read_text())
    queue['schema_version']=5;queue['settings'].update(queue_path='configs/experiments/fm_crossad_complete_queue.v5.json',
        preflight_parent_queue=parent.relative_to(ROOT).as_posix(),preflight_parent_sha256=sha(parent))
    queue['worker_script']='run_crossad_case_gated_queue.py'
    common=['scripts/flow_matching/register_crossad_case_gated.py','scripts/flow_matching/run_crossad_case_gated_queue.py']
    for job in queue['jobs']:
        assert not (ROOT/job['output_directory']/'workspace').exists()
        job['frozen_files'] += [{'path':p,'sha256':sha(ROOT/p)} for p in common]
    queue['superseded_unlaunched_queues'].append(parent.relative_to(ROOT).as_posix())
    queue['boundary']+=' Full native training-batch proof is required separately for each dataset/component case. Ungated cases stay pending in the complete 139-job scope; no budget or architecture is substituted.'
    target=ROOT/queue['settings']['queue_path']
    if target.exists() and json.loads(target.read_text())!=queue:raise ValueError('Case-gated queue frozen')
    write_json(target,queue)
    write_json(ROOT/'docs/reports/fm_crossad_case_gated_registration_2026-10-09.json',{'registered_jobs':139,
        'queue_path':target.relative_to(ROOT).as_posix(),'queue_sha256':sha(target),'preflight_parent':str(parent.relative_to(ROOT)).replace(chr(92),'/'),
        'preflight_parent_sha256':sha(parent),'boundary':queue['boundary'],'all_experiments_complete':False})
    print('Registered all 139 case-gated full-budget jobs',sha(target))


if __name__=='__main__':main()
