"""Invoke original frozen implementations with only fresh job identity/output changed."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--queue',required=True);parser.add_argument('--job-id',required=True);args=parser.parse_args()
    queue=json.loads((ROOT/args.queue).read_text()); case=next(c for c in queue['jobs'] if c['job']['id']==args.job_id)
    assert sha(ROOT/case['original_queue'])==case['original_queue_sha256']
    job,original=case['job'],case['original_job']
    assert {k:v for k,v in job.items() if k not in ('id','output_directory')}=={k:v for k,v in original.items() if k not in ('id','output_directory')}
    for record in job['frozen_files']:
        assert sha(ROOT/record['path'])==record['sha256'],record['path']
    if case['kind']=='pi':
        from scripts.flow_matching.run_pi_transformer_job import run
        run(job,case['settings'])
    elif case['kind']=='maelnet':
        from scripts.flow_matching.run_maelnet_author_queue import run_job
        state={'pid':__import__('os').getpid(),'active_job':job['id']}
        run_job(job,case['settings'],state,ROOT/job['output_directory']/'author_stage_status.json')
    elif case['kind']=='strict':
        from scripts.flow_matching.run_tsad_experiment import run
        run(ROOT,job,case['evaluation'])
    elif case['kind']=='industrial':
        from scripts.flow_matching.run_industrial_tsad_experiment import run
        run(ROOT,job,case['evaluation'])
    else:raise ValueError('Unknown recovery kind')


if __name__=='__main__':main()
