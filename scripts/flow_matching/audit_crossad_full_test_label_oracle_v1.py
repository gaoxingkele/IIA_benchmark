"""Complete fixed-score, test-label-selected F1 upper bounds;not a benchmark.

Exact tie-group sweeps enumerate every strict-threshold prediction set. PA uses
the true-event maximum score, which exactly fills hit true events at any cutoff.
"""
import argparse
import copy
import json
import math
from pathlib import Path
from types import FunctionType

from scripts.flow_matching import audit_crossad_same_score_protocol_factorial_v1 as source
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

MODULE='scripts.flow_matching.audit_crossad_full_test_label_oracle_v1'
CONFIG='configs/reproducibility/fm_crossad_full_test_label_oracle.v2.json'
TEST='tests/test_fm_crossad_full_test_label_oracle_v1.py'


def pa_scores(scores,labels):
    if len(scores)!=len(labels) or any(y not in (0,1) for y in labels):raise ValueError('Full aligned binary labels required')
    result=list(scores);first=0
    while first<len(labels):
        if not labels[first]:first+=1;continue
        last=first+1
        while last<len(labels) and labels[last]:last+=1
        maximum=max(scores[first:last]);result[first:last]=[maximum]*(last-first);first=last
    return result


def exact_best_f1(scores,labels):
    if (not scores or len(scores)!=len(labels) or any(y not in (0,1) for y in labels)
            or not all(math.isfinite(x) for x in scores)):
        raise ValueError('Every finite full score and aligned binary label required')
    pairs=sorted(zip(scores,labels),reverse=True)
    positives=sum(labels);tp=fp=0
    best=dict(threshold=max(scores),true_positives=0,false_positives=0,false_negatives=positives,
        precision=0.,recall=0.,f1=0.)
    numerator,denominator=0,1
    index=0;groups=0
    while index<len(pairs):
        score=pairs[index][0];last=index
        while last<len(pairs) and pairs[last][0]==score:
            tp+=int(pairs[last][1]);fp+=int(not pairs[last][1]);last+=1
        groups+=1;fn=positives-tp;num=2*tp;den=2*tp+fp+fn
        # Integer cross-products select the true best ratio, with conservative
        # highest-threshold tie breaking;no floating comparison changes ties.
        if den and num*denominator>numerator*den:
            numerator,denominator=num,den
            best=dict(threshold=math.nextafter(score,-math.inf),true_positives=tp,false_positives=fp,false_negatives=fn,
                precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/positives if positives else 0.,f1=num/den)
        index=last
    return dict(**best,evaluated_full_test_points=len(scores),complete_unique_score_groups=groups,
        evaluated_strict_threshold_prediction_sets=groups+1,threshold_operator='strict_greater',
        tie_break='highest cutoff among exact equal F1 values',test_labels_used_to_select_threshold=True,
        diagnostic_only=True,deployable_validation_only_result=False)


def oracle(scores,labels):
    raw=exact_best_f1(scores,labels)
    adjusted=exact_best_f1(pa_scores(scores,labels),labels)
    for row,uses_pa in [(raw,False),(adjusted,True)]:
        row['point_adjustment']=uses_pa;row['test_labels_used_for_event_postprocessing']=uses_pa
    # Reconstruct the full prediction population at each selected threshold,
    # independently of the sweep counts,including all normal false alarms.
    for row in [raw,adjusted]:
        predictions=[v>row['threshold'] for v in scores]
        if row['point_adjustment']:predictions=source.point_adjust(labels,predictions)
        counts=source.confusion(labels,predictions)
        for key in counts:
            if counts[key]!=row[key]:raise ValueError('Exact full-population oracle sweep/reconstruction mismatch')
    return dict(raw=raw,PA=adjusted)


def registration():
    c=copy.deepcopy(source.registration())
    c.update(state_root='experiments/runs/fm_crossad_full_test_label_oracle_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/crossad_full_test_label_oracle_v1',
        original_same_score_diagnostic_configuration=source.CONFIG,
        posthoc_test_label_selected_diagnostic=True,prospectively_registered_benchmark=False,
        exact_threshold_search='Every complete test score tie group;integer F1 cross-product comparison;highest-cutoff tie breaking',
        oracle_protocols=['raw','PA'],point_adjustment_uses_true_event_labels=True,
        test_label_threshold_selection=True,original_fixed_calibration_controls_test_label_threshold_selection=False,
        superseded_unexecuted_metadata_draft='configs/reproducibility/fm_crossad_full_test_label_oracle.v1.json',
        oracle_scores_never_enter_strict_leaderboards=True)
    pins={x['path']:x['sha256'] for x in c['source_receipts']}
    for p in [source.CONFIG,MODULE.replace('.','/')+'.py',TEST,c['superseded_unexecuted_metadata_draft']]:pins[p]=sha(ROOT/p)
    c['source_receipts']=[dict(path=p,sha256=h) for p,h in sorted(pins.items())]
    return c


def verify(c):
    if c!=registration():raise ValueError('Frozen all139 full-test diagnostic and exact tie-sweep contract required')
    return source.verify(read(ROOT/source.CONFIG))


def audit(c,q,upstream,job):
    observed=[]
    def with_oracle(train,validation,test,labels,budgets):
        observed.append(oracle(test,labels))
        return source.factorial(train,validation,test,labels,budgets)
    f=FunctionType(source.audit.__code__,dict(source.audit.__globals__,factorial=with_oracle))
    result=f(c,q,upstream,job)
    if len(observed)!=1:raise ValueError('Exactly one complete original test population required')
    result.update(full_test_label_selected_upper_bounds=observed[0],
        posthoc_test_label_selected_diagnostic=True,prospectively_registered_benchmark=False,
        oracle_scores_never_enter_strict_leaderboards=True)
    return result


def cohort_summary(q,verified):
    value=source.cohort_summary(q,verified)
    for key,group in value['cohorts'].items():
        if not group['complete']:continue
        records={j['seed']:verified[j['canonical_CPU_condition_reference']] for j in q['jobs']
            if j['mode']=='fresh_train' and j['seed'] in source.groups.SEEDS
            and source.groups.cohort_key(j)==key}
        group['test_label_selected_F1_upper_bounds']={p:source.groups.interval([
            records[seed]['full_test_label_selected_upper_bounds'][p]['f1'] for seed in source.groups.SEEDS]) for p in ['raw','PA']}
        group['paired_oracle_PA_minus_validation_1percent_raw']=source.groups.interval([
            records[seed]['full_test_label_selected_upper_bounds']['PA']['f1']
            -records[seed]['protocol_factorial']['1']['protocols']['validation_raw']['f1'] for seed in source.groups.SEEDS])
    value.update(oracle_thresholds_selected_using_test_labels=True,oracle_scores_never_enter_strict_leaderboards=True,
        posthoc_analysis_not_model_superiority=True)
    return value


def main():
    # The unchanged guarded per-case controller retains all139 original models.
    def write_summary(path,value):
        write(path,value)
        config=ROOT/CONFIG
        if config.exists():
            c=read(config)
            if Path(path)==ROOT/c['state_root']/'status.json' and 'verified_ids' in value:
                q=read(ROOT/c['queue'])
                records={i:read(Path(c['artifact_root'])/'conditions'/(i+'.json')) for i in value['verified_ids']}
                write(Path(c['artifact_root'])/'all22_oracle_diagnostic_cohorts.json',cohort_summary(q,records))
    f=FunctionType(source.parent.main.__code__,dict(source.parent.main.__globals__,
        MODULE=MODULE,CONFIG=CONFIG,registration=registration,verify=verify,audit=audit,write=write_summary))
    f()


if __name__=='__main__':main()
