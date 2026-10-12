"""Execute one complete released SHCL command with observed, unchanged math."""
import argparse
from pathlib import Path

from scripts.flow_matching import shcl_native_observer_v1 as observer
from scripts.flow_matching.shcl_author_audits_v1 import environment


def verify(queue):
    for pin in queue['source_receipts']:
        if observer.native.sha(observer.native.ROOT/pin['path'])!=pin['sha256']:
            raise ValueError('Frozen full SHCL script execution changed: '+pin['path'])
    model=observer.native.read(observer.native.ROOT/queue['model_config']); observer.native.verify(model)
    if (len(queue['jobs'])!=413 or sum(j['mode']=='train' for j in queue['jobs'])!=49
            or sum(j['mode']=='test' for j in queue['jobs'])!=364):
        raise ValueError('All49 source trainings and364 tests required')
    registry=observer.native.read(observer.native.ROOT/queue['script_registry'])
    files={p['path']:p['sha256'] for job in queue['jobs'] for p in job['full_data_files']}
    for path,digest in files.items():
        if observer.native.sha(path)!=digest: raise ValueError('Full prepared original input changed')
    for index,(job,source) in enumerate(zip(queue['jobs'],registry['commands'])):
        if job['source_command_index']!=index or len(job['argv'])!=len(source['argv']) or job['native_DataLoader_workers']!=8:
            raise ValueError('Every source command and original worker count required')
        for i,value in enumerate(source['argv']):
            if i and source['argv'][i-1] in ('--data_path','--model_save_path'): continue
            if job['argv'][i]!=value: raise ValueError('Original non-filesystem command tokens changed')
    return model


def worker(queue,job):
    model=verify(queue)
    if environment()!=queue['environment']: raise ValueError('Actual frozen native numerical environment required')
    if job['depends_on']:
        dependency=next(j for j in queue['jobs'] if j['id']==job['depends_on'])
        result=observer.native.read(Path(dependency['output_directory'])/'observer/result.json')
        if (not result['original_training_complete'] or result['optimizer_updates']!=dependency['expected_optimizer_updates']
                or len(result['epochs'])!=dependency['num_epochs']):
            raise ValueError('Full registered original training checkpoint prerequisite required')
        path=Path(job['checkpoint_directory'])/(job['dataset']+'_checkpoint.pth')
        if observer.native.sha(path)!=result['epochs'][-1]['original_model_checkpoint_sha256']:
            raise ValueError('Actual last native checkpoint differs from full training receipt')
    base=Path(job['output_directory']); base.mkdir(parents=True,exist_ok=False)
    target=base/'observer'; target.mkdir()
    Path(job['checkpoint_directory']).parent.mkdir(parents=True,exist_ok=True)
    recording=observer.Observer(job,target); observer.install(model,recording)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('Registered native CUDA execution required; do not silently fall back to CPU')
    observer.native.execute_author_command(model,job['argv'],base/'native')
    result=observer.native.read(target/'result.json')
    if result['job']!=job or not result['passed']: raise ValueError('Complete registered native command receipt required')


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--queue',required=True); parser.add_argument('--job-id',required=True)
    args=parser.parse_args(); queue=observer.native.read(observer.native.ROOT/args.queue)
    job=next(j for j in queue['jobs'] if j['id']==args.job_id); worker(queue,job)


if __name__=='__main__':main()
