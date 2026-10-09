"""Audit full GRASP cohorts; incomplete entities never become paper macro results."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math

from scripts.flow_matching.grasp_protocol_runner import ROOT, complete, sha, verify, write_json


def aggregate(values):
    import numpy as np
    from scipy.stats import t
    if not values:
        return {'n':0,'mean':None,'std':None,'se':None,'ci95_low':None,'ci95_high':None}
    mean = float(np.mean(values))
    std = float(np.std(values,ddof=1)) if len(values)>1 else None
    se = std/math.sqrt(len(values)) if std is not None else None
    delta = float(t.ppf(.975,len(values)-1))*se if se is not None else None
    return {'n':len(values),'mean':mean,'std':std,'se':se,
            'ci95_low':mean-delta if delta is not None else None,'ci95_high':mean+delta if delta is not None else None}


def summarize(queue, root=ROOT):
    import psutil
    manifest = verify(queue,root)
    state_path = root / queue['state_root'] / 'status.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    live = False
    if state.get('active_pid'):
        try:
            command = psutil.Process(state['active_pid']).cmdline()
            live = 'scripts.flow_matching.grasp_protocol_runner' in command and state.get('active_job') in command
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            pass
    records, results = [], {}
    for job in queue['jobs']:
        path = root / job['output_directory']
        done = complete(job,root)
        record = {'id':job['id'],'profile':job['profile'],'dataset':job['dataset'],'entity':job['entity'],'seed':job['seed'],
                  'status':'completed' if done else 'running' if live and state.get('active_job')==job['id']
                  else 'failed_or_partial_preserved' if path.exists() else 'pending'}
        if done:
            result_path = path/'result.json'
            result = json.loads(result_path.read_text(encoding='utf-8'))
            results[job['id']] = result
            record['result_path'], record['result_sha256'] = result_path.relative_to(root).as_posix(),sha(result_path)
        records.append(record)
    lookup = {(j['profile'],j['dataset'],j['entity'],j['seed']):j for j in queue['jobs']}
    cohorts = []
    for profile in queue['profiles']:
        for data in manifest['datasets']:
            for seed in queue['seeds']:
                jobs = [lookup[profile,data['id'],entity['id'],seed] for entity in data['entities']]
                available = [results[j['id']] for j in jobs if j['id'] in results]
                ready = len(available)==len(jobs)
                cohort = {'profile':profile,'dataset':data['id'],'seed':seed,'completed_entities':len(available),
                          'required_entities':len(jobs),'status':'completed' if ready else 'incomplete','metrics':None}
                if ready:
                    recipes = jobs[0]['score_recipes']
                    metrics = []
                    for recipe in recipes:
                        per_entity = [next(m for m in r['metrics'] if m['recipe']==recipe) for r in available]
                        values = {key:[r['metrics']['paper_retrospective'][key] for r in per_entity
                                       if r['metrics']['paper_retrospective'][key] is not None] for key in ['AP','ROC','Best_F1']}
                        values.update({key:[r['metrics']['validation_only_control'][key] for r in per_entity]
                                       for key in ['precision','recall','f1']})
                        metrics.append({'recipe':recipe,'entity_macro':{k:aggregate(v) for k,v in values.items()},
                                        'aggregation':'unweighted entity macro; undefined ROC explicitly excluded',
                                        'test_score_seconds':sum(m['test_score_seconds'] for m in per_entity)})
                    cohort['metrics'] = metrics
                cohorts.append(cohort)
    seeds = []
    for profile in queue['profiles']:
        for data in manifest['datasets']:
            ready = [c for c in cohorts if c['profile']==profile and c['dataset']==data['id'] and c['status']=='completed']
            metrics = []
            if ready:
                for recipe in ready[0]['metrics']:
                    scores = [next(m for m in c['metrics'] if m['recipe']==recipe['recipe']) for c in ready]
                    keys = scores[0]['entity_macro']
                    metrics.append({'recipe':recipe['recipe'],'ten_seed_metrics':{k:aggregate([s['entity_macro'][k]['mean'] for s in scores
                                                    if s['entity_macro'][k]['mean'] is not None]) for k in keys}})
            seeds.append({'profile':profile,'dataset':data['id'],'completed_seeds':len(ready),
                          'required_seeds':len(queue['seeds']),'metrics':metrics if ready else None})
    return {'captured_utc':datetime.now(timezone.utc).isoformat(),'entity_job_counts':dict(Counter(r['status'] for r in records)),
            'registered_entity_jobs':len(records),'registered_cohorts':len(cohorts),
            'completed_cohorts':sum(c['status']=='completed' for c in cohorts),'jobs':records,'cohorts':cohorts,
            'seed_aggregates':seeds,'active_process_verified_live':live,'controller_state':state,
            'paper_equivalence_certified':False,'all_direction_experiments_complete':False,'boundary':queue['boundary']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',default='configs/experiments/fm_grasp_protocol_queue.v2.json')
    parser.add_argument('--output',default='docs/reports/fm_grasp_protocol_progress_2026-10-09.json')
    args = parser.parse_args()
    queue = json.loads((ROOT/args.queue).read_text(encoding='utf-8'))
    report = summarize(queue)
    report.update(queue=args.queue,queue_sha256=sha(ROOT/args.queue))
    write_json(ROOT/args.output,report)
    print(json.dumps({k:report[k] for k in ['entity_job_counts','registered_entity_jobs','registered_cohorts','completed_cohorts','active_process_verified_live']}))


if __name__ == '__main__':
    main()
