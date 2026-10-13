"""Audit every paired full GP400/GP600 Sine outcome without rerunning models.

All five original seeds and every held-out grid remain required. Preserve the
two original GP600 scoring failures and their complete80-trajectory diagnostics.
The five-seed GP600 mean stays undefined. CFM-TS: OpenReview Hqn4Aj7xrQ.
"""
import argparse
from datetime import datetime,timezone
from pathlib import Path
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold
from scripts.flow_matching.audit_cfm_full_arrays_v1 import audit_arrays

ROOT,read,sha=cold.ROOT,cold.original.read,cold.streaming_sha


def complete_pair_summary(rows):
    if len(rows)!=5 or sorted(r['seed'] for r in rows)!=list(range(1103,1108)):
        raise ValueError('All five distinct original paired seeds are required')
    successes=[r for r in rows if r['GP600_original_scoring_complete']]
    return dict(required_paired_seeds=5,GP400_complete_original_scores=5,
        GP600_complete_original_scores=len(successes),GP600_failed_original_scores=5-len(successes),
        GP600_full_five_seed_MSE_mean=None if len(successes)!=5 else sum(r['GP600_MSE'] for r in successes)/5,
        failure_free_five_seed_GP600_performance_complete=len(successes)==5,
        complete_success_subset_promoted_as_full_cohort=False)


def audit(config):
    import numpy as np
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen paired evidence changed: '+pin['path'])
    queue=read(ROOT/config['original_queue']);jobs={j['id']:j for j in queue['jobs']}
    runtime=read(ROOT/config['replay_runtime']);verified=read(ROOT/config['verified_snapshot'])
    if len(jobs)!=285 or runtime['original_scope']!=config['original_scope']:
        raise ValueError('Full original research scope and all285 CFM jobs remain registered')
    certs={r['original_canonical_id']:r for r in verified['records']}
    models=[read(ROOT/config[key]) for key in ['GP400_model','GP600_model']]
    changed=[k for k in models[0]['parameters'] if models[0]['parameters'][k]!=models[1]['parameters'][k]]
    if changed!=['epochs'] or set(models[0]['parameters'])!=set(models[1]['parameters']):
        raise ValueError('Paired source recipes must differ only in epoch budget')
    from scripts.flow_matching import audit_cfm_gp600_seed1103_full80_failure_v2 as GP3
    from scripts.flow_matching import audit_cfm_gp600_seed1105_full80_failure_v1 as GP5
    failed={1103:(GP3,config['GP1103_post_configuration']),1105:(GP5,config['GP1105_post_configuration'])}
    failed_reports={}
    for seed,(module,path) in failed.items():
        post=read(ROOT/path);fresh=module.post_audit(post);stored=read(ROOT/post['output_report'])
        for key in fresh:
            if key!='captured_utc' and fresh[key]!=stored[key]:raise ValueError('Full failed-grid independent proof changed')
        failed_reports[seed]=stored
    rows=[];coordinate_checks=0
    for seed in range(1103,1108):
        ids=['cfm_ts_original__appendix_d__gp_paper_literal_e'+str(e)+'__sine__seed'+str(seed) for e in [400,600]]
        pair=[jobs[i] for i in ids]
        if (pair[0]['data_path']!=pair[1]['data_path'] or pair[0]['data_sha256']!=pair[1]['data_sha256']
                or pair[0]['model_overrides']!=pair[1]['model_overrides']
                or any(j['method']!='gp' or j['test_trajectories']!=80 or j['test_points_per_trajectory']!=500 for j in pair)
                or [j['model_config'] for j in pair]!=[config['GP400_model'],config['GP600_model']]
                or [j['expected_optimizer_updates'] for j in pair]!=[250000,375000]
                or sha(ROOT/pair[0]['data_path'])!=pair[0]['data_sha256']):
            raise ValueError('Exact same full original data, seed, model overrides and both complete budgets required')
        row=dict(seed=seed,data_sha256=pair[0]['data_sha256'],same_full_original_data_and_split=True,
            GP400_canonical_id=ids[0],GP600_canonical_id=ids[1],GP600_original_scoring_complete=seed not in failed)
        results=[]
        for index,job in enumerate(pair):
            if index==1 and seed in failed:continue
            resultpath=ROOT/job['output_directory']/'result.json';result=read(resultpath)
            source=read(ROOT/'experiments/runs/fm_cfm_ts_original_v1/receipts'/(job['id']+'.json'))
            if result['job']!=job or result['status']!='completed' or not source['complete'] or source['exit_code']!=0 or source['resource_observations']['resource_guard_aborted']:
                raise ValueError('Actual complete original guarded source score required')
            for artifact in result['artifacts']:
                if sha(artifact['path'])!=artifact['sha256']:raise ValueError('Complete original trained artifact changed')
            predictions=next(a['path'] for a in result['artifacts'] if Path(a['path']).name=='predictions.npz')
            with np.load(predictions,allow_pickle=False) as arrays,np.load(ROOT/job['data_path'],allow_pickle=False) as data:
                metrics=audit_arrays(job,result,arrays,data)
                if set(data['train_ids'])&set(data['test_ids']):raise ValueError('Whole-trajectory split must be disjoint')
                coordinate_checks+=arrays['predictions'].size
                row['GP'+str(job['epochs'])+'_MSE']=metrics['MSE']
                if index==0 and seed in failed:
                    failure=failed_reports[seed]
                    details=[]
                    for evidence in failure['failure_evidence']:
                        position=int(np.flatnonzero(arrays['test_ids']==evidence['trajectory_id'])[0])
                        prediction=arrays['predictions'][position];truth=arrays['truth'][position]
                        details.append(dict(trajectory_id=evidence['trajectory_id'],
                            GP400_full_grid_MSE=float(np.square(prediction.astype(np.float64)-truth).mean()),
                            GP400_full_grid_finite=bool(np.isfinite(prediction).all()),
                            GP600_error=evidence['error_message'],GP600_initial_state=evidence['original_initial_state'],
                            GP600_initial_inside_training_range=evidence['initial_state_inside_observed_training_state_range']))
                    row['paired_previously_finite_GP400_now_failed_GP600_trajectories']=details
            row['GP'+str(job['epochs'])+'_result_sha256']=sha(resultpath);results.append(result)
            if index==1:cold.validate_closed_primary_record(runtime,certs[job['id']])
        if seed in failed:
            failure=failed_reports[seed];row.update(GP600_MSE=None,
                GP600_full80_failure_observation_complete=True,GP600_finite_trajectories=failure['finite_full_grid_trajectories'],
                GP600_failed_trajectory_ids=failure['failed_trajectory_ids'],
                GP600_full_model_budget_verified_by_unchanged_native_worker=True,
                GP600_loss_history_prefix_equality=None)
        else:
            row['GP600_loss_history_prefix_equality']=results[0]['training_losses']==results[1]['training_losses'][:400]
        rows.append(row)
    return dict(captured_utc=datetime.now(timezone.utc).isoformat(),original_scope=config['original_scope'],**complete_pair_summary(rows),records=rows,
        independent_successful_grid_coordinates_checked=coordinate_checks,
        independent_failed_diagnostic_grid_coordinates_checked=sum(r['independent_checked_grid_coordinates'] for r in failed_reports.values()),
        only_source_parameter_difference='epochs:400 -> 600',
        same_seed_successful_model_first400_recorded_losses_equal=all(r['GP600_loss_history_prefix_equality'] for r in rows if r['GP600_original_scoring_complete']),
        all_failed_full_grid_diagnostics_preserved=True,additional_ODE_calls=0,additional_optimizer_updates=0,
        test_based_recipe_selection=False,training_root_cause_or_author_bug_certified=False,
        author_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False,
        diagnosis_boundary='The same original files yield five finite GP400 scores and two GP600 numerical failures. This does not identify the causal training mechanism or justify selecting budgets from test outcomes.')


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=read(ROOT/args.config);result=audit(config)
    destination=ROOT/config['output_report']
    with destination.open('x',encoding='utf-8') as stream:
        import json
        json.dump(result,stream,indent=2,allow_nan=False)
    print({k:result[k] for k in ['GP400_complete_original_scores','GP600_complete_original_scores','GP600_failed_original_scores','GP600_full_five_seed_MSE_mean','independent_successful_grid_coordinates_checked','independent_failed_diagnostic_grid_coordinates_checked']})


if __name__=='__main__':main()
