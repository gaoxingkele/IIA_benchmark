"""Freeze all139 original CrossAD executions, including90 reconstructed ablations."""
import copy
from datetime import datetime,timezone
import json
from pathlib import Path

from scripts.flow_matching.run_crossad_completion_replay_queue_v1 import ROOT,read,digest,verify,GUARDS,SCORE_TOLERANCE,METRIC_TOLERANCE
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json


def main():
    template_path='configs/reproducibility/fm_crossad_full_native_msl_model_replay.2026-10-11_v2.json'
    template=read(template_path)
    runtime={key:copy.deepcopy(template[key]) for key in (*GUARDS,'original_scope')}
    runtime.update(original_queue='configs/experiments/fm_crossad_complete_queue.v7.json',native_replay_template=template_path,
        required_original_jobs=139,required_original_fresh_training_jobs=132,required_original_fixed_release_jobs=7,
        required_original_reconstructed_ablation_jobs=90,author_manifest='configs/experiments/fm_crossad_author_execution.v1.json',
        score_tolerance=SCORE_TOLERANCE,metric_tolerance=METRIC_TOLERANCE,poll_seconds=10,
        state_root='experiments/runs/fm_crossad_completion_replay_queue_v1',
        child_state_root='experiments/runs/fm_crossad_completion_replay_children_v1',
        configuration_root='configs/reproducibility/fm_crossad_completion_replay_children_v1',
        proof_root='projects/flow_matching_research/results/2026-10-12/crossad_full139_completion_replay_v1/completed_models',
        artifact_root='F:/aicoding/IIA_Data/experiments/crossad_full139_completion_replay_v1',preserved_verified_jobs=[])
    sources={p['path'] for p in template['source_receipts']}
    sources.update([template_path,runtime['original_queue'],runtime['author_manifest'],
        'scripts/flow_matching/register_crossad_completion_replay_queue_v1.py',
        'scripts/flow_matching/run_crossad_completion_replay_queue_v1.py','scripts/flow_matching/replay_crossad_registered_model_v3.py',
        'scripts/flow_matching/replay_crossad_full_native_model_v2.py','scripts/flow_matching/audit_crossad_full_release_completion_v1.py',
        'src/iia_benchmark/models/fm_crossad_ablation.py','scripts/flow_matching/run_crossad_ablation_v2_job.py',
        'tests/test_fm_crossad_completion_replay_queue_v1.py'])
    queue=read(runtime['original_queue'])
    for job in queue['jobs']:sources.update(pin['path'] for pin in job['frozen_files'])
    for path in [template_path,'configs/reproducibility/fm_crossad_full_release_msl_model_replay.2026-10-11_v1.json',
                 'configs/reproducibility/fm_crossad_full_release_gecco_model_replay.2026-10-11_v1.json']:
        config=read(path)
        runtime['preserved_verified_jobs'].append(dict(job_id=config['job_id'],native_configuration=path))
        sources.update(pin['path'] for pin in config['source_receipts'])
        sources.update([path,config['independent_point_completion'],config['state_root']+'/receipt.json',
            str(Path(config['artifact_directory'])/'full_native_model_replay.json')])
        job=next(j for j in queue['jobs'] if j['id']==config['job_id'])
        sources.add(str(Path(job['output_directory'])/'result.json'))
        sources.update(artifact['path'] for artifact in read(ROOT/job['output_directory']/'result.json')['artifacts'])
    runtime['source_receipts']=[dict(path=p,sha256=digest(p)) for p in sorted(sources)]
    jobs,_,covered=verify(runtime)
    ready=[job['id'] for job in queue['jobs'] if job['id'] not in covered
        and (ROOT/job['output_directory']/'result.json').exists()
        and read(ROOT/job['output_directory']/'result.json').get('status')=='completed']
    path='configs/runtime/fm_crossad_completion_replay_queue.v1.json'
    exclusive_json(ROOT/path,runtime)
    evidence=ROOT/'projects/flow_matching_research/results/2026-10-12/crossad_full139_completion_replay_v1'
    exclusive_json(evidence/'registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=path,configuration_sha256=digest(path),required_original_jobs=139,required_original_fresh_training_jobs=132,
        required_original_fixed_release_jobs=7,required_original_reconstructed_ablation_jobs=90,
        preserved_complete_native_and_independent_jobs=len(covered),pending_original_jobs=139-len(covered),
        currently_complete_but_unreviewed_jobs=ready,source_pins=len(sources),original_scope=runtime['original_scope'],
        numeric_publication_unchanged=True,author_equivalence_certified=False,
        unpublished_author_ablation_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False))
    print(json.dumps(dict(configuration=path,required_original_jobs=len(jobs),preserved_verified_jobs=len(covered),
        pending_original_jobs=len(jobs)-len(covered),complete_but_unreviewed_jobs=ready,source_pins=len(sources))))


if __name__=='__main__':main()
