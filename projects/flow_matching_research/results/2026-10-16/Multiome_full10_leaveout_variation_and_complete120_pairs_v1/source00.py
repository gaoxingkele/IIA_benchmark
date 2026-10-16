"""All120conditions:complete leaveout/seed diagnostics and matched method pairs."""
import argparse
import ast
import inspect as pyinspect
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive
from scripts.flow_matching import summarize_otcfm_Multiome_full120_paper_rows_v1 as rows

MODULE='scripts.flow_matching.analyze_otcfm_Multiome_complete120_comparisons_v1'
CONFIG='configs/reproducibility/fm_otcfm_Multiome_complete120_comparison_diagnostics.v1.json'
TEST='tests/test_fm_otcfm_Multiome_complete120_comparisons_v1.py'
METRICS=('1-Wasserstein','2-Wasserstein','Mean_MSE','Mean_L2','Mean_L1','Median_MSE','Median_L2','Median_L1')
CRITICAL=2.7764451051977987


def five(values):
    if len(values)!=5 or not all(math.isfinite(v) for v in values):raise ValueError('All five finite original training seeds required')
    mean=math.fsum(values)/5;std=statistics.stdev(values);se=std/math.sqrt(5)
    return dict(n=5,mean=mean,sample_STD=std,se=se,df=4,ci95=[mean-CRITICAL*se,mean+CRITICAL*se])


def balanced(records,group):
    if (len(records)!=10 or {r['id'] for r in records}!=set(group['required_case_ids'])
            or len({r['id'] for r in records})!=10
            or {(r['seed'],r['leaveout_timepoint']) for r in records}!=set(itertools.product(range(42,47),[1,2]))
            or any((r['dataset'],r['protocol'],r['method'])!=(group['dataset'],group['protocol'],group['method']) for r in records)
            or any(set(r['heldout'])!=set(METRICS) or any(not math.isfinite(v) or v<0 for v in r['heldout'].values()) for r in records)):
        raise ValueError('One complete balanced10model input/recipe/method is required;no missing/mixed/pseudorepeated conditions')
    matrix={(r['seed'],r['leaveout_timepoint']):r['heldout'] for r in records};out={}
    for metric in METRICS:
        values=[matrix[seed,left][metric] for seed,left in itertools.product(range(42,47),[1,2])]
        seedmeans=[math.fsum(matrix[seed,left][metric] for left in [1,2])/2 for seed in range(42,47)]
        leftmeans={left:math.fsum(matrix[seed,left][metric] for seed in range(42,47))/5 for left in [1,2]}
        grand=math.fsum(values)/10
        total=math.fsum((v-grand)**2 for v in values)
        leftSS=5*math.fsum((m-grand)**2 for m in leftmeans.values())
        seedSS=2*math.fsum((m-grand)**2 for m in seedmeans)
        residual=math.fsum((matrix[seed,left][metric]-leftmeans[left]-seedmeans[seed-42]+grand)**2
            for seed,left in itertools.product(range(42,47),[1,2]))
        if not math.isclose(total,leftSS+seedSS+residual,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Balanced descriptive sum-of-squares identity must hold')
        out[metric]=dict(seed_level_two_leaveout_means=seedmeans,primary_five_seed_summary=five(seedmeans),
            per_leaveout_five_seed_summary={str(left):five([matrix[seed,left][metric] for seed in range(42,47)]) for left in [1,2]},
            paired_leaveout2_minus1=five([matrix[seed,2][metric]-matrix[seed,1][metric] for seed in range(42,47)]),
            all10_model_sample_STD=statistics.stdev(values),
            descriptive_variance_decomposition=dict(total_SS=total,leaveout_SS=leftSS,seed_SS=seedSS,residual_SS=residual,
                leaveout_fraction=leftSS/total if total else None,seed_fraction=seedSS/total if total else None,
                residual_fraction=residual/total if total else None,causal_or_ANOVA_hypothesis_test=False))
    return out


def paired(left,right,leftgroup,rightgroup):
    balanced(left,leftgroup);balanced(right,rightgroup)
    if leftgroup['dataset']!=rightgroup['dataset'] or leftgroup['protocol']!=rightgroup['protocol'] or leftgroup['method']==rightgroup['method']:
        raise ValueError('Pair two distinct methods on one unchanged input and scientific recipe')
    a={(r['seed'],r['leaveout_timepoint']):r['heldout'] for r in left}
    b={(r['seed'],r['leaveout_timepoint']):r['heldout'] for r in right}
    return {metric:dict(methodA=leftgroup['method'],methodB=rightgroup['method'],negative_difference_favors_methodB=True,
        per_seed_two_leaveout_mean_B_minus_A=[math.fsum(b[seed,l][metric]-a[seed,l][metric] for l in [1,2])/2 for seed in range(42,47)],
        paired_five_seed_summary=five([math.fsum(b[seed,l][metric]-a[seed,l][metric] for l in [1,2])/2 for seed in range(42,47)])) for metric in METRICS}


def registration():
    rc=read(ROOT/rows.CONFIG);rows.verify(rc);q=read(ROOT/rows.native.QUEUE);groups=[]
    for g in rc['groups']:
        groups.append(dict(id='within__'+g['id'],kind='within',upstream_row_ids=[g['id']],required_case_ids=g['required_case_ids']))
    for dataset,protocol in itertools.product(sorted({g['dataset'] for g in rc['groups']}),q['protocol_tracks']):
        selected=[g for g in rc['groups'] if g['dataset']==dataset and g['protocol']==protocol]
        for a,b in itertools.combinations(selected,2):
            groups.append(dict(id='paired__'+a['id']+'__vs__'+b['method'],kind='paired',upstream_row_ids=[a['id'],b['id']],
                required_case_ids=a['required_case_ids']+b['required_case_ids']))
    ids=[i for g in groups for i in g['required_case_ids']]
    if len(groups)!=24 or len(set(ids))!=120 or len(ids)!=360:raise ValueError('All12within and12matched pairs over the same120models required')
    pins=[rows.CONFIG,rows.MODULE.replace('.','/')+'.py',MODULE.replace('.','/')+'.py',TEST,rows.native.QUEUE,rows.replay.CONFIG,
        'scripts/flow_matching/otcfm_singlecell_result_integrity_v2.py',rows.generic.template.MODULE.replace('.','/')+'.py']
    return dict(groups=groups,required_diagnostics=24,required_distinct_original_models=120,model_condition_references=360,
        upstream_complete_rows=rows.CONFIG,source_queue=rows.native.QUEUE,native_replay_configuration=rows.replay.CONFIG,
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in pins],metrics=list(METRICS),training_seeds=list(range(42,47)),
        leaveout_indices=[1,2],settings=rc['settings'],resource_lock=rc['resource_lock'],python=sys.executable,numerical_threads=2,
        state_root='experiments/runs/fm_otcfm_Multiome_complete120_comparison_diagnostics_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/otcfm_Multiome_complete120_comparison_diagnostics_v1',
        posthoc_diagnostics=True,SD_convention_selected_to_match_paper=False,hypothesis_testing_or_causal_claim=False,
        original_scope=rc['original_scope'],citation=rc['paper_citation'],all_original_experiments_complete=False,
        boundary='All120actual original model conditions,12within-row diagnostics and12within-input/recipe method pairs. '
            'Gate on actual complete upstream10model row guards,not only model guards,to avoid racing row aggregation. '
            'All8actual heldout metrics remain;five seed-level averages are the uncertainty unit,not ten models or evaluator repeats. '
            'Balanced leaveout/seed/residual SS is descriptive,not causal attribution or an ANOVA significance test. '
            'Pair every method pair on same source data/split/net/scientific recipe and matched original seeds/leaveouts. '
            'Actual stopping/update/time differences are disclosed,not presumed compute-matched. '
            'No inference,training,LP rerun,closest-to-paper SD selection,partial-row promotion or multiplicity/generalization claim. '
            'Historical identity and original validation/full-population/whitening boundaries remain;all420defined singlecell and45paper obligations remain.')


def verify(c):
    if c!=registration():raise ValueError('All120conditions,24diagnostics,8metrics and original resources must remain')


def upstream(gid):
    c=read(ROOT/rows.CONFIG);p=Path(c['artifact_root'])/gid/'complete_paper_row.json'
    gp=ROOT/c['state_root']/'attempts'/gid/'guard_receipt.json'
    if not gp.exists():return None
    try:g=read(gp)
    except json.JSONDecodeError:return None
    if not g['passed'] or g['exit_code']!=0 or g['resource_guard_aborted']:
        return dict(phase='upstream_complete_row',failed_case=gid,preserved_guard_path=str(gp))
    if (not p.exists() or g['report_sha256']!=sha(p) or not read(p)['passed']
            or read(p)['configuration_sha256']!=sha(ROOT/rows.CONFIG)):
        raise ValueError('Passed upstream full-row guard must bind its immutable complete report')
    return p,gp


def availability(q,rc,job):
    raw=rows.generic.availability(q,rc,job)
    if raw is not True:return raw
    result=upstream(job['dataset']+'__'+job['protocol']+'__'+job['method'])
    return True if isinstance(result,tuple) else result


def worker(c,group):
    q=read(ROOT/rows.native.QUEUE);rc=read(ROOT/rows.replay.CONFIG);cfg=read(ROOT/rows.CONFIG);byrow={};evidence=[];budgets=[]
    for gid in group['upstream_row_ids']:
        paths=upstream(gid)
        if not isinstance(paths,tuple):raise ValueError('Complete guarded upstream row is required before any diagnostics')
        report=read(paths[0]);g=next(g for g in cfg['groups'] if g['id']==gid);records=[]
        for identity in g['required_case_ids']:
            job=next(j for j in q['jobs'] if j['id']==identity);source,proof=rows.inspect(q,job)
            _,ng,rp=rows.generic.guard_paths(q,rc,job);native,guard=read(rp),read(ng)
            rows.validate_native(job,source,proof,native,guard,sha(rp),sha(ROOT/rows.replay.CONFIG))
            old=next(p for p in report['proofs'] if p['id']==identity)
            if old['source_provenance']['result_sha256']!=proof['result_sha256'] or old['native_report_sha256']!=sha(rp):
                raise ValueError('Every current source/native proof must match the completed upstream row')
            for a in native['all_full_test_arrays_and_artifacts']:
                p=Path(a['path'])
                if p.stat().st_size!=a['bytes'] or sha(p)!=a['sha256']:raise ValueError('Original replay artifacts changed')
            records.append(dict(id=identity,dataset=job['dataset'],protocol=job['protocol'],method=job['method'],seed=job['seed'],
                leaveout_timepoint=job['leaveout_timepoint'],heldout={m:source['test_metrics']['test/t_out/'+m] for m in METRICS}))
            budgets.append(dict(id=identity,actual_epochs=source['actual_completed_epochs'],actual_updates=source['epoch_history'][-1]['optimizer_updates'],
                selected_updates=source['optimizer_updates'],training_seconds=source['training_seconds'],termination_reason=source['termination_reason']))
        byrow[gid]=(g,records);evidence.append(dict(upstream_row_id=gid,report_sha256=sha(paths[0]),guard_sha256=sha(paths[1])))
    if group['kind']=='within':
        g,r=byrow[group['upstream_row_ids'][0]];result=balanced(r,g)
    else:
        a,b=[byrow[i] for i in group['upstream_row_ids']]
        for seed,left in itertools.product(range(42,47),[1,2]):
            ja,jb=[next(j for j in q['jobs'] if j['id'] in g['required_case_ids'] and j['seed']==seed and j['leaveout_timepoint']==left) for g,_ in [a,b]]
            if any(ja[k]!=jb[k] for k in ['data','scientific_protocol','net','expected_model_parameters','native_dimensions']):
                raise ValueError('Every paired method condition must retain same input/split/recipe/architecture')
        result=paired(a[1],b[1],a[0],b[0])
    out=Path(c['artifact_root'])/group['id'];out.mkdir(parents=True,exist_ok=False)
    exclusive(out/'comparison_diagnostic.json',dict(passed=True,id=group['id'],kind=group['kind'],metrics=result,
        source_rows={gid:r for gid,(_,r) in byrow.items()},upstream_evidence=evidence,actual_budgets=budgets,
        configuration_sha256=sha(ROOT/CONFIG),posthoc=True,extra_training_or_inference=False,
        causal_or_hypothesis_test_claim=False,historical_paper_equivalence_certified=False,all_original_experiments_complete=False))


def controller(c):
    template=rows.generic.template;tree=ast.parse(pyinspect.getsource(template.controller))
    replacements={'completed_rows':'completed_diagnostics','failed_rows_preserved':'failed_diagnostics_preserved',
        'available_complete_native_cases_per_row':'available_complete_native_cases_per_diagnostic',
        'waiting_all15_source_and_native_conditions_per_row':'waiting_all_original_models_and_complete_row_guards',
        'all6_paper_row_outcomes_terminal_other_original_work_remains':'all24_diagnostic_outcomes_terminal_other_original_work_remains',
        'complete_paper_row.json':'comparison_diagnostic.json'}
    class Scope(ast.NodeTransformer):
        def visit_keyword(self,n):
            self.generic_visit(n)
            n.arg=replacements.get(n.arg,n.arg)
            return n
        def visit_Constant(self,n):
            if type(n.value) is int and n.value==6:n.value=24
            elif isinstance(n.value,str):n.value=replacements.get(n.value,n.value)
            return n
    namespace=dict(template.controller.__globals__,MODULE=MODULE,CONFIG=CONFIG,native=rows.native,replay=rows.replay,
        worker=worker,availability=availability)
    exec(compile(ast.fix_missing_locations(Scope().visit(tree)),__file__,'exec'),namespace);namespace['controller'](c)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--group-id');a=p.parse_args()
    if a.register:exclusive(ROOT/CONFIG,registration());return
    c=read(ROOT/CONFIG);verify(c)
    if a.group_id:worker(c,next(g for g in c['groups'] if g['id']==a.group_id))
    else:controller(c)


if __name__=='__main__':main()
