"""Compare complete15-model EB rows only after all native/full-W1 proofs pass."""
import argparse
from datetime import datetime,timezone
import itertools
import json
import math
from pathlib import Path
import os
import subprocess
import sys
import time
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive
from scripts.flow_matching import run_otcfm_EB100_code_referenced_native90_v1 as native
from scripts.flow_matching import audit_otcfm_EB100_full90_replay_v1 as replay
from scripts.flow_matching.otcfm_EB100_result_integrity_v1 import inspect

MODULE='scripts.flow_matching.summarize_otcfm_EB100_full_paper_rows_v1'
CONFIG='configs/reproducibility/fm_otcfm_EB100_all90_full_paper_rows.v1.json'
TEST='tests/test_fm_otcfm_EB100_full_paper_rows_v1.py'
TARGETS={'cfm':dict(paper_name='I-CFM',mean=.872,std=.087),
         'otcfm':dict(paper_name='OT-CFM',mean=.790,std=.068),
         'sbcfm':dict(paper_name='SB-CFM',mean=1.221,std=.380)}


def registration():
    q=read(ROOT/native.QUEUE);native.verify(q);r=read(ROOT/replay.CONFIG);replay.verify(r)
    groups=[]
    for protocol,method in itertools.product(native.contract.PROTOCOLS,native.contract.METHODS):
        jobs=[j for j in q['jobs'] if j['protocol']==protocol and j['method']==method]
        if len(jobs)!=15 or {(j['seed'],j['leaveout_timepoint']) for j in jobs}!=set(itertools.product(range(42,47),[1,2,3])):
            raise ValueError('Every original EB method row requires all five seeds and all three separately trained leaveouts')
        groups.append(dict(id=protocol+'__'+method,protocol=protocol,method=method,
            required_case_ids=[j['id'] for j in jobs],paper_target=TARGETS[method]))
    policy_path='configs/reproducibility/fm_common105_matched577_all1726_labels.2026-10-16_v1.json'
    policy=read(ROOT/policy_path)
    paths=[native.QUEUE,replay.CONFIG,MODULE.replace('.','/')+'.py',TEST,policy_path,
        'scripts/flow_matching/otcfm_EB100_result_integrity_v1.py','scripts/flow_matching/audit_cfm_full_arrays_v1.py',
        'papers/literature/flow_matching/pdfs/ot_cfm.pdf','papers/literature/flow_matching/extracted/ot_cfm.txt']
    return dict(groups=groups,required_rows=6,required_distinct_native_models=90,source_queue=native.QUEUE,
        native_replay_configuration=replay.CONFIG,source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in paths],
        paper_citation=q['citation'],paper_target_table=4,paper_PDF_page_one_based=11,paper_setup_section='E.5',
        mean_aggregation='equal mean of three separately trained heldout models within each seed,then equal mean over five seeds',
        seed_uncertainty='sampleSTD/t95df4 over the five seed-level three-leaveout means',
        alternative_spread='sampleSTD of all15separately trained heldout W1 values,reported separately without assuming15independent seeds',
        author_reported_SD_aggregation_explicitly_certified=False,all_15_models_and_native_replays_required=True,
        no_table_result_from_partial_rows=True,no_test_based_recipe_or_label_selection=True,
        settings=policy['settings'],resource_lock=policy['resource_lock'],numerical_threads=2,python=sys.executable,
        state_root='experiments/runs/fm_otcfm_EB100_all90_full_paper_rows_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/otcfm_EB100_all90_full_paper_rows_v1',
        original_scope=q['original_scope'],all_original_experiments_complete=False,
        boundary='Metadata-only complete-row aggregation,not extra training,inference or LP execution. '
            'All6method/recipe rows require15original conditions and all full native44scalar/three-timepoint LP proofs. '
            'Use actual heldout W1,never the source mean across interior times in one model. '
            'Both recipes and both disclosed spread definitions remain separate;no closest-to-paper selection. '
            'Published Table4 SD aggregation and historical hyperparameter/weight identity remain uncertified. '
            'Use existing readonly scalar audit policy;all original native32GiB guards and complete420-defined singlecell/all45paper scope preserved.')


def verify(c):
    if c!=registration():raise ValueError('All6complete paper-row comparisons and all90source/full-native proofs required')


def summarize(records,method,protocol):
    import statistics
    from scripts.flow_matching.audit_cfm_full_arrays_v1 import independent_stats
    expected=set(itertools.product(range(42,47),[1,2,3]))
    if (len(records)!=15 or len({r['id'] for r in records})!=15
            or {(r['seed'],r['leaveout_timepoint']) for r in records}!=expected
            or any(r['method']!=method or r['protocol']!=protocol for r in records)
            or not all(math.isfinite(r['heldout_W1']) and r['heldout_W1']>=0 for r in records)):
        raise ValueError('Do not publish paper rows from missing/duplicated/mixed methods,recipes,seeds or leaveouts')
    seed_means=[math.fsum(r['heldout_W1'] for r in records if r['seed']==seed)/3 for seed in range(42,47)]
    values=[r['heldout_W1'] for r in records]
    stats=independent_stats(seed_means)
    if not math.isclose(stats['mean'],math.fsum(values)/15,rel_tol=1e-14,abs_tol=1e-14):raise ValueError('Every original leaveout must receive equal weight')
    return dict(method=method,protocol=protocol,independently_trained_models=15,training_seeds=5,leaveouts_per_seed=3,
        seed_level_leaveout_means=seed_means,seed_mean_summary=stats,degrees_of_freedom=4,
        all15_model_sample_STD=statistics.stdev(values),paper_target=TARGETS[method],
        mean_difference_from_paper=stats['mean']-TARGETS[method]['mean'],
        author_reported_SD_aggregation_certified=False,historical_paper_equivalence_certified=False,
        comparison='FULL15_SOURCE_AND_NATIVE_COMPLETE_NUMERIC_COMPARISON_ONLY',all_original_experiments_complete=False)


def guard_paths(q,r,job):
    key=Path(job['artifact_directory']).name
    return (ROOT/q['state_root']/'attempts'/key/'guard_receipt.json',
        ROOT/r['state_root']/'attempts'/key/'guard_receipt.json',Path(r['artifact_root'])/key/'full_checkpoint_replay.json')


def availability(q,r,job):
    source_guard,replay_guard,report=guard_paths(q,r,job)
    for phase,path in [('source',source_guard),('native_replay',replay_guard)]:
        if not path.exists():return None
        try:g=read(path)
        except json.JSONDecodeError:return None  # A producer can still be committing its terminal JSON.
        if not g['passed'] or g['exit_code']!=0 or g['resource_guard_aborted']:
            return dict(phase=phase,failed_case=job['id'],preserved_guard_path=str(path),preserved_guard_sha256=sha(path))
    return True if report.exists() else None


def worker(c,group):
    q=read(ROOT/native.QUEUE);rc=read(ROOT/replay.CONFIG);records=[];proofs=[]
    for identity in group['required_case_ids']:
        job=next(j for j in q['jobs'] if j['id']==identity)
        if availability(q,rc,job) is not True:raise ValueError('Every actual source and native replay must finish before row aggregation')
        source,provenance=inspect(q,job);sg,ng,rp=guard_paths(q,rc,job);r,g=read(rp),read(ng)
        if (not g['passed'] or g['full_replay_report_sha256']!=sha(rp) or r['status']!='passed' or r['id']!=identity
                or r['configuration_sha256']!=sha(ROOT/replay.CONFIG)
                or r['source_provenance']['result_sha256']!=provenance['result_sha256']
                or r['original_metrics']!=source['test_metrics'] or not r['full_native_test_reinference_passed']
                or not r['all_learned_tensors_and_optimizer_moments_checked']
                or r['native_metric_tolerance']!=dict(rtol=2e-6,atol=2e-6) or len(r['native_metrics'])!=44):
            raise ValueError('Every full original source and all44metric native checkpoint proof required')
        for name,value in r['native_metrics'].items():
            if not math.isclose(value,source['test_metrics'][name],rel_tol=2e-6,abs_tol=2e-6):raise ValueError('Original full-metric tolerances remain')
        solves=r['full_test_population_distances_independently_recomputed'][0]['solves']
        if len(solves)!=6 or any(s['native_LP_result_code']!=1 for s in solves):raise ValueError('All three complete EB W1/W2 LP solves required')
        for s in solves:
            if s['power']==1 and (s['independent_LP_result_code']!=1 or not s['full_costs_independently_recomputed']
                    or s['objective_absolute_difference']>s['LP_objective_perturbation_bound']):raise ValueError('Every full independent W1 solve/bound required')
        for item in r['all_full_test_arrays_and_artifacts']:
            p=Path(item['path'])
            if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('Every actual replay array and artifact must remain immutable')
        records.append(dict(id=identity,method=job['method'],protocol=job['protocol'],seed=job['seed'],
            leaveout_timepoint=job['leaveout_timepoint'],heldout_W1=source['primary_heldout_W1']))
        proofs.append(dict(id=identity,source_provenance=provenance,replay_report_sha256=sha(rp),replay_guard_sha256=sha(ng)))
    summary=summarize(records,group['method'],group['protocol'])
    out=Path(c['artifact_root'])/group['id'];out.mkdir(parents=True,exist_ok=False)
    exclusive(out/'complete_paper_row.json',dict(passed=True,**summary,records=records,proofs=proofs,
        configuration_sha256=sha(ROOT/CONFIG),source_array_model_or_metric_rewrites=False,extra_training_or_inference=False))


def controller(c):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=install_facades();base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Complete EB paper-row collector already owned')
    if (base/'status.json').exists():owner.close();raise FileExistsError('Preserve earlier row attempts')
    q=read(ROOT/native.QUEUE);rc=read(ROOT/replay.CONFIG);completed=[];failed={}
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,all_original_experiments_complete=False)
    def update(v):state.update(v,completed_rows=completed,failed_rows_preserved=failed,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        while len(completed)+len(failed)<6:
            counts={}
            for group in c['groups']:
                if group['id'] in completed or group['id'] in failed:continue
                states=[availability(q,rc,next(j for j in q['jobs'] if j['id']==identity)) for identity in group['required_case_ids']]
                counts[group['id']]=sum(s is True for s in states)
                source_failures=[s for s in states if isinstance(s,dict)]
                if source_failures:failed[group['id']]=source_failures;continue
                if not all(s is True for s in states):continue
                attempt=base/'attempts'/group['id'];out=Path(c['artifact_root'])/group['id']
                try:
                    if attempt.exists() or out.exists():raise FileExistsError('Preserve all full/partial group attempts without retry')
                    with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],update):
                        attempt.mkdir(parents=True,exist_ok=False)
                        command=[c['python'],'-B','-X','utf8','-u','-m',MODULE,'--group-id',group['id']]
                        with (attempt/'stdout.log').open('xb') as stdout,(attempt/'stderr.log').open('xb') as stderr:
                            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                            handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
                            update(dict(state='auditing_complete15_model_paper_row',active_pid=child.pid,active_group=group['id'],exact_child_handle=handle))
                            code,usage=capped_wait(child,c,api,update)
                        report=out/'complete_paper_row.json';passed=code==0 and not usage['resource_guard_aborted'] and report.exists() and read(report)['passed']
                        exclusive(attempt/'guard_receipt.json',dict(passed=passed,exit_code=code,**usage,exact_child_handle=handle,report_sha256=sha(report) if report.exists() else None))
                        if not passed:raise ValueError('Full row metadata audit failed;all partials preserved')
                        completed.append(group['id'])
                except Exception as error:failed[group['id']]=str(error)
            update(dict(state='waiting_all15_source_and_native_conditions_per_row',available_complete_native_cases_per_row=counts,active_pid=None));time.sleep(10)
        update(dict(state='all6_paper_row_outcomes_terminal_other_original_work_remains',active_pid=None))
    finally:owner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true');parser.add_argument('--group-id');a=parser.parse_args()
    if a.register:exclusive(ROOT/CONFIG,registration());return
    c=read(ROOT/CONFIG);verify(c)
    if a.group_id:worker(c,next(g for g in c['groups'] if g['id']==a.group_id))
    else:controller(c)


if __name__=='__main__':main()
