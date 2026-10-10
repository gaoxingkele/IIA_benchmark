"""Register original stable industrial jobs blocked by empty preserved outputs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from scripts.flow_matching.run_light_controller import ROOT,digest


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();path=ROOT/args.config;config=read(path)
    queue_path=ROOT/config['original_queue'];original=read(queue_path)
    if digest(queue_path)!=config['original_queue_sha256']:raise ValueError('Original industrial scientific queue differs')
    cases=[]
    for identity in config['original_ids']:
        job=next(j for j in original['jobs'] if j['id']==identity)
        output=ROOT/job['output_directory']
        if not output.is_dir() or any(output.iterdir()):raise ValueError('Registered original empty output evidence differs')
        if 'stable_v2' not in Path(job['model_config']).stem:raise ValueError('This exact recovery selects registered stable original configurations only')
        replacement=copy.deepcopy(job);replacement['id']+='__empty_output_recovery_v1'
        replacement['output_directory']=config['state_root']+'/jobs/'+replacement['id']
        if (ROOT/replacement['output_directory']).exists():raise FileExistsError('Preserve previous exact recovery output')
        if {k:v for k,v in replacement.items() if k not in {'id','output_directory'}}!={k:v for k,v in job.items() if k not in {'id','output_directory'}}:
            raise ValueError('Original full experiment changed')
        for entry in job['frozen_files']:
            if digest(ROOT/entry['path'])!=entry['sha256']:raise ValueError('Original full frozen source changed')
        cases.append(dict(kind='industrial',original_queue=config['original_queue'],original_queue_sha256=digest(queue_path),
            original_id=job['id'],original_job=job,job=replacement,settings=original.get('settings',{}),
            evaluation=original['evaluation'],python=original['python'],
            evidence=dict(original_empty_output_directory=job['output_directory'],old_directory_preserved=True,
                cause='Original worker refuses any existing output directory; empty directories are not evidence of training'),
            failure_record=dict(status='empty_output_blocked_before_training')))
    frozen=[path,Path(__file__),ROOT/'scripts/flow_matching/run_resource_recovery.py',
        ROOT/'scripts/flow_matching/run_resource_recovery_job.py',ROOT/'scripts/flow_matching/run_industrial_tsad_experiment.py']
    recovery=dict(schema_version=3,settings=dict(config['settings'],output_root=config['state_root'],queue_path=config['output_queue']),
        jobs=cases,frozen_files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=digest(p)) for p in frozen],
        prior_registered_recovery_queues=config['prior_registered_recovery_queues'],
        boundary='Nine original stable industrial slots; only attempt ID and output directory differ. Same full data, model, parameters, evaluation, seed and original source. Empty old directories preserved. Original complete preferred, otherwise first valid registered recovery; no extra seed or metric selection. The75 old numerically unstable configurations remain explicitly unresolved and are not declared reproduced by this registration.')
    destination=ROOT/config['output_queue']
    if destination.exists():raise FileExistsError('Frozen exact recovery queue already exists')
    destination.write_text(json.dumps(recovery,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(registered_full_original_jobs=len(cases),queue=config['output_queue'],sha256=digest(destination))))


if __name__=='__main__':main()
