"""Freeze the missing complete tau1/seed1104 replay without changing active experiments."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.flow_matching.run_grasp_registered_replay_review_pipeline_v1 import ROOT,read,sha,exclusive_json,binding


def main():
    canonical='grasp_protocol_v3__tau_1__swan__swan_sf__seed1104'
    queue_path='configs/experiments/fm_grasp_protocol_queue.v4.json';queue=read(ROOT/queue_path)
    job=next(j for j in queue['jobs'] if j['id']==canonical)
    result_path=(Path(job['output_directory'])/'result.json').as_posix();result=read(ROOT/result_path)
    if (result['status']!='completed' or result['job']!=job or result['epochs_completed']!=1500
            or result['optimizer_updates']!=job['expected_optimizer_updates']
            or result['validation_points']!=job['validation_points'] or result['scored_points']!=job['test_points']
            or result['score_recipes']!=job['score_recipes'] or result['diagnostic']
            or result['paper_equivalence_certified'] or result['strict_TAB_result']):
        raise ValueError('Actual completed original tau1 full-budget model required')
    template_path='configs/reproducibility/fm_grasp_tau051104_checkpoint_replay.2026-10-11_v1.json'
    template=read(ROOT/template_path)
    sources={p['path'] for p in template['source_receipts']}
    sources.update([template_path,queue_path,result_path,job['model_config'],queue['environment_lock'],
        result['data_audit']['prepared_path'],'scripts/flow_matching/register_grasp_tau1_seed1104_full_review_v1.py',
        'scripts/flow_matching/run_grasp_registered_replay_review_pipeline_v1.py',
        'scripts/flow_matching/verify_grasp_completed_full_replay_v1.py',
        'scripts/flow_matching/audit_grasp_checkpoint_replay_resume_v2.py',
        'tests/test_fm_grasp_registered_replay_review_pipeline_v1.py',
        'tests/test_fm_grasp_completed_full_replay_v1.py'])
    for artifact in result['artifacts']+result['data_audit']['raw_sources']:
        if sha(ROOT/artifact['path'])!=artifact['sha256']:raise ValueError('Preserve exact original native sources/artifacts')
        sources.add(artifact['path'])
    receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(sources)]
    evidence='projects/flow_matching_research/results/2026-10-12/grasp_tau11104_full_replay_review_v1'
    producer=copy.deepcopy(template)
    producer.update(full_results=[dict(canonical_id=canonical,result_path=result_path,result_sha256=sha(ROOT/result_path))],
        output_directory=evidence+'/native',state_root='experiments/runs/fm_grasp_tau11104_full_checkpoint_replay_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/grasp_tau11104_full_checkpoint_replay_v1',
        full_job_count=1,full_score_recipe_count=len(result['metrics']),source_receipts=receipts,
        boundary='Original completed tau1/SWAN/swan_sf/seed1104 full1500epochs/93000updates; '
            'all3935validation/72000test points and original5source x10flow recipe. Unchanged native full '
            'checkpoint replay/solver/environment/score/metric tolerances and resource guards. Preserve '
            'all9120GRASP entity jobs/480cohorts/10original repeats and45paper scope. No retraining, '
            'shortened grid, added seed, retrospective threshold promotion or original-author/TAB certification.')
    producer_path='configs/reproducibility/fm_grasp_tau11104_checkpoint_replay.2026-10-12_v1.json'
    exclusive_json(ROOT/producer_path,producer)
    pipeline={k:copy.deepcopy(producer[k]) for k in ['python','cpu_threads','settings','resource_lock','score_tolerance','metric_tolerance','original_scope']}
    pipeline.update(producer_configuration=producer_path,canonical_id=canonical,
        review_configuration='configs/reproducibility/fm_grasp_tau11104_completed_full_replay_review.2026-10-12_v1.json',
        review_output_directory=evidence+'/independent',output_directory=evidence+'/completion',
        state_root='experiments/runs/fm_grasp_tau11104_full_replay_review_pipeline_v1',
        source_receipts=receipts+[dict(path=producer_path,sha256=sha(ROOT/producer_path))],
        boundary=producer['boundary']+' Require actual complete native terminal receipt before independent '
            'full-array/forward-count/recipe/metric verification. Published performance table stays frozen until '
            'the complete original evidence and its appropriate uncertainty audit are independently accepted.')
    pipeline_path='configs/reproducibility/fm_grasp_tau11104_replay_review_pipeline.2026-10-12_v1.json'
    exclusive_json(ROOT/pipeline_path,pipeline);binding(pipeline)
    target=ROOT/evidence;target.mkdir(parents=True,exist_ok=False)
    exclusive_json(target/'registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        canonical_id=canonical,epochs_completed=1500,optimizer_updates=result['optimizer_updates'],
        validation_points=result['validation_points'],test_points=result['scored_points'],score_recipes=result['score_recipes'],
        producer_configuration=producer_path,pipeline_configuration=pipeline_path,
        original_scope=pipeline['original_scope'],full_registered_GRASP_jobs=9120,cohorts=480,
        original_full_training_already_completed=True,full_model_replay_completed=False,
        independent_full_replay_review_completed=False,published_numeric_table_unchanged=True,
        additional_training_seed_slots=0,all_original_experiments_complete=False))
    paths=[producer_path,pipeline_path,'scripts/flow_matching/register_grasp_tau1_seed1104_full_review_v1.py',
        'scripts/flow_matching/run_grasp_registered_replay_review_pipeline_v1.py',
        'tests/test_fm_grasp_registered_replay_review_pipeline_v1.py']
    exclusive_json(ROOT/'tmp/grasp_tau11104_full_replay_review_v1_paths.json',dict(
        paths=paths,evidence=evidence,producer=producer_path,pipeline=pipeline_path,
        previous_root='fff5b9f98301c4d5297449573f90d9202cc9c4b6',previous_parent='32ca7af77e21e891c5bb81811699d14a713d00c1'))
    print(json.dumps(dict(canonical_id=canonical,full_original_epochs=1500,full_original_updates=result['optimizer_updates'],
        full_original_test_points=result['scored_points'],full_recipes=len(result['metrics']),source_pins=len(sources),
        pipeline=pipeline_path,full_model_replay_completed=False)))


if __name__=='__main__':main()
