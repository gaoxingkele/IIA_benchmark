"""Read-only complete epoch/frame replay for one canonical full source command."""
import argparse
from pathlib import Path

from scripts.flow_matching import replay_shcl_native_frames_v1 as replay
from scripts.flow_matching.run_shcl_native_job_v1 import verify as verify_queue
from scripts.flow_matching.run_shcl_native_script_queue_v1 import complete
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def verify(config):
    for pin in config['source_receipts']:
        if replay.native.sha(replay.native.ROOT/pin['path'])!=pin['sha256']:
            raise ValueError('Frozen complete source replay changed: '+pin['path'])
    queue=replay.native.read(replay.native.ROOT/config['original_queue']); model=verify_queue(queue)
    if config['canonical_job_ids']!=[j['id'] for j in queue['jobs']] or len(config['canonical_job_ids'])!=413:
        raise ValueError('Every original49train/364test canonical command must remain in scope')
    return queue,model


def worker(config,job_id):
    queue,model=verify(config)
    if replay.environment()!=queue['environment']: raise ValueError('Same actual original native environment required')
    import torch
    torch.set_num_threads(2)
    job=next(j for j in queue['jobs'] if j['id']==job_id)
    if not complete(job): raise ValueError('Actual complete original source command required before read-only replay')
    output=Path(config['output_root'])/job_id
    if output.exists(): raise FileExistsError('Preserve every full source replay attempt')
    output.mkdir(parents=True)
    source_result=replay.native.read(Path(job['output_directory'])/'observer/result.json')
    checkpoint=Path(job['checkpoint_directory'])/(job['dataset']+'_checkpoint.pth')
    report=dict(passed=False,canonical_job=job,original_source_result_sha256=replay.native.sha(
        Path(job['output_directory'])/'observer/result.json'),checkpoint_sha256=replay.native.sha(checkpoint),
        original_scope=queue['original_scope'],original_source_command_count=413,original_training_commands=49,
        original_test_commands=364,new_optimizer_updates_executed=0,new_benchmark_results=0,
        original_paper_equivalence_certified=False,all_original_experiments_complete=False)
    if job['mode']=='train':
        replay.replay_training(model,source_result,checkpoint,report)
    else:
        replay.replay_test(model,source_result,checkpoint,output/'complete_replayed_arrays',report)
    report['passed']=True
    exclusive(output/'full_native_source_replay.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--config',required=True); parser.add_argument('--job-id',required=True)
    args=parser.parse_args(); worker(replay.native.read(replay.native.ROOT/args.config),args.job_id)


if __name__=='__main__':main()
