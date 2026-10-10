"""Register exact LS4 canonical attempts for the verified fallback correction."""
import argparse
import copy
import json
from pathlib import Path

from scripts.flow_matching.run_ls4_cauchy_corrected_v3 import ROOT,sha,verify
from scripts.flow_matching.giflow_native_protocol import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();settings=json.loads((ROOT/args.config).read_text())
    for family in settings['families']:
        original_path=family['original_queue'];old=json.loads((ROOT/original_path).read_text())
        path=ROOT/family['queue_path']
        if path.exists():raise FileExistsError('Preserve frozen corrected LS4 registration')
        queue=copy.deepcopy(old);queue.update(canonical_family=family['name'],queue_path=family['queue_path'],
            original_canonical_queue=original_path,original_canonical_queue_sha256=sha(ROOT/original_path),
            state_root=family['state_root'],cauchy_correction_config=settings['cauchy_correction_config'],
            boundary=old['boundary']+' Verified naive Cauchy conjugate correction in a separately named full canonical execution. All original data,model kwargs,seeds and full budgets retained; raw sources untouched. Original author backend equivalence unproved.')
        artifact=Path(settings['artifact_root'])/family['name']
        for name in ['data_audit','gpu_preflight']:
            queue[name+'_artifact']=str(artifact/name)
            queue[name+'_output']=family['state_root']+'/preflight/'+name+'.json'
        if 'data_audit_progress' in old:
            queue['data_audit_progress']=family['state_root']+'/preflight/full_data_audit_progress.json'
        paths={args.config,original_path,settings['cauchy_correction_config'],
            'scripts/flow_matching/ls4_cauchy_fallback_repair_v1.py',
            'scripts/flow_matching/run_ls4_cauchy_corrected_v3.py',
            'scripts/flow_matching/run_ls4_cauchy_corrected_queue_v3.py',
            'scripts/flow_matching/register_ls4_cauchy_corrected_v3.py',
            'tests/test_fm_ls4_cauchy_corrected_v3.py',
            'scripts/flow_matching/transition_ls4_cauchy_idle_v3.py'}
        paths.update(settings['paired_diagnostic_sources'])
        for job,previous in zip(queue['jobs'],old['jobs']):
            model=json.loads((ROOT/job['model_config']).read_text())
            model.update(implementation='scripts.flow_matching.run_ls4_cauchy_corrected_v3',
                reproduction_status='full_native_canonical_attempt_registered_official_Cauchy_correction_not_completed',
                author_equivalence_certified=False,runtime_backend_correction=settings['cauchy_correction_config'],
                original_model_config=previous['model_config'])
            new_model='configs/models/fm_ls4_cauchy_corrected_v3__'+family['name']+'__'+job['id']+'.json'
            if (ROOT/new_model).exists():raise FileExistsError('Preserve corrected model config')
            write_json(ROOT/new_model,model);paths.add(new_model)
            job.update(model_config=new_model,model_config_sha256=sha(ROOT/new_model),
                original_output_directory=previous['output_directory'],
                original_artifact_directory=previous['artifact_directory'],
                output_directory=family['state_root']+'/jobs/'+job['id'],artifact_directory=str(artifact/job['id']))
        queue['source_receipts'] += [dict(path=p,sha256=sha(ROOT/p)) for p in sorted(paths)]
        verify(queue);write_json(path,queue)
        print(json.dumps(dict(family=family['name'],original_seed_slots=len(queue['jobs']),
            queue_sha256=sha(path),generator_updates=sum(json.loads((ROOT/j['model_config']).read_text())['budget']['generator_updates'] for j in queue['jobs']))))


if __name__=='__main__':main()
