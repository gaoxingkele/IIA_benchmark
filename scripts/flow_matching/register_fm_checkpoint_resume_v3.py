"""Freeze completed replay evidence and register only the original remaining scope."""
import copy
from pathlib import Path

import psutil

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT,read,sha,write


def receipt(config,path):
    item=dict(path=str(path),sha256=sha(ROOT/path))
    if item not in config['source_receipts']:
        config['source_receipts'].append(item)


def terminal(config):
    state_path=Path(config['state_root'])/'status.json'
    state=read(ROOT/state_path)
    if state['state']!='failed_preserved':
        raise ValueError('Only an authoritative preserved terminal attempt can resume')
    for key,time_key in [('pid','create_time'),('active_pid','active_create_time')]:
        if state.get(key):
            try:
                if abs(psutil.Process(state[key]).create_time()-state[time_key])<.02:
                    raise RuntimeError('Original replay handle remains live')
            except psutil.NoSuchProcess:
                pass
    return state_path


def register_grasp():
    previous='configs/reproducibility/fm_grasp_full_checkpoint_replay.2026-10-11_v2.json'
    config=copy.deepcopy(read(ROOT/previous));terminal_path=terminal(config)
    output=ROOT/config['output_directory'];completed=[]
    for registered in config['full_results']:
        path=output/(registered['canonical_id']+'.json')
        if not path.exists():
            continue
        partial=read(path);result=read(ROOT/registered['result_path'])
        recipes=partial.get('completed_recipes',[])
        expected=result['job']['score_recipes']
        if len(recipes)!=len(expected):
            continue
        if ({(r['recipe']['source_samples'],r['recipe']['flow_evaluations']) for r in recipes}
                !={(r['source_samples'],r['flow_evaluations']) for r in expected}):
            raise ValueError('Complete original scoring recipe coverage required')
        for record in recipes:
            if (not record['full_recipe_reinference_passed']
                    or record['complete_test_points']!=result['job']['test_points']
                    or record['complete_validation_points']!=result['job']['validation_points']):
                raise ValueError('Only full passed native checkpoint recipes can be preserved')
            for field in ['replay_score','original_score']:
                if sha(record[field+'_path'])!=record[field+'_sha256']:
                    raise ValueError('Full passed recipe arrays changed')
                receipt(config,record[field+'_path'])
        receipt(config,path.relative_to(ROOT))
        completed.append(dict(canonical_id=registered['canonical_id'],recipe_count=len(recipes),
            path=path.relative_to(ROOT).as_posix(),sha256=sha(path),scientific_result=registered))
    assert len(completed)==2
    completed_ids={r['canonical_id'] for r in completed}
    config['full_results']=[r for r in config['full_results'] if r['canonical_id'] not in completed_ids]
    config['full_job_count']=len(config['full_results'])
    config['full_score_recipe_count']=sum(len(read(ROOT/r['result_path'])['job']['score_recipes']) for r in config['full_results'])
    config['completed_full_replays_preserved']+=completed
    assert config['full_job_count']==6 and len(config['completed_full_replays_preserved'])==7
    for path in [previous,terminal_path,Path(config['state_root'])/'receipt.json',Path(config['output_directory'])/'progress.json']:
        receipt(config,path)
    config.update(output_directory='projects/flow_matching_research/results/2026-10-11/grasp_full_checkpoint_replay_v3',
        artifact_root='F:/aicoding/IIA_Data/experiments/grasp_full_checkpoint_replay_v3',
        state_root='experiments/runs/fm_grasp_full_checkpoint_replay_v3')
    config['settings']['minimum_commit_headroom_bytes']=42*2**30
    config['boundary']+=' Successor after the second authoritative emergency-memory abort. All7 passed original model replays are frozen; all6 remaining original checkpoints and24 full scoring recipes remain required. The newly trained tau0.25 model is separately fully replayed, not counted as an additional seed here. Startup commit gate increases to42GiB; emergency16GiB and scientific budgets/tolerances unchanged.'
    path=ROOT/'configs/reproducibility/fm_grasp_full_checkpoint_replay.2026-10-11_v3.json'
    if path.exists():raise FileExistsError(path)
    write(path,config)
    return path


def register_giflow():
    previous='configs/reproducibility/fm_giflow_full_checkpoint_replay.2026-10-11_v1.json'
    config=copy.deepcopy(read(ROOT/previous));terminal_path=terminal(config)
    progress_path=Path(config['output_directory'])/'progress.json';progress=read(ROOT/progress_path)
    assert progress['completed_batches']==9 and progress['forward_calls']==180
    batches=[]
    for index in range(progress['completed_batches']):
        path=Path(config['artifact_root'])/f'batch_{index:05d}.npz'
        batches.append(dict(batch=index,path=str(path),sha256=sha(path)))
        receipt(config,path)
    config['preserved_prefix']=dict(batch_count=len(batches),batches=batches,progress_path=progress_path.as_posix())
    for path in [previous,terminal_path,Path(config['state_root'])/'receipt.json',progress_path,
                 'scripts/flow_matching/audit_giflow_checkpoint_replay_resume_v2.py',
                 'tests/test_fm_giflow_checkpoint_replay_resume_v2.py']:
        receipt(config,path)
    config.update(output_directory='projects/flow_matching_research/results/2026-10-11/giflow_full_checkpoint_replay_v2',
        artifact_root='F:/aicoding/IIA_Data/experiments/giflow_full_checkpoint_replay_v2',
        state_root='experiments/runs/fm_giflow_full_checkpoint_replay_v2')
    config['settings']['minimum_commit_headroom_bytes']=42*2**30
    config['boundary']+=' Terminal v1 emergency-memory abort preserved9 complete native CPU-replayed batches. Each prefix cache must match freshly loaded original targets/masks/positions/shapes and GPU original predictions. Resume all5 remaining batches with the identical original20step sampler; seed the original metric lists with native float32 prefix reductions, retaining all14 original unweighted batch means. Preserve full1747window/268613target/280forward budget across attempts; no new training seed. Startup commit gate42GiB; emergency16GiB unchanged.'
    path=ROOT/'configs/reproducibility/fm_giflow_full_checkpoint_replay.2026-10-11_v2.json'
    if path.exists():raise FileExistsError(path)
    write(path,config)
    return path


def main():
    for path in [register_grasp(),register_giflow()]:
        print(path.relative_to(ROOT).as_posix(),sha(path))


if __name__=='__main__':
    main()
