"""Full matched577 fixed-score test-label-selected diagnostics, never rankings."""
import argparse
import ast
import copy
import json
from pathlib import Path
from types import FunctionType
from scripts.flow_matching import audit_common105_validation_calibration_v1 as upstream
from scripts.flow_matching import audit_crossad_full_test_label_oracle_v1 as exact

ROOT,read,sha,write=upstream.ROOT,upstream.read,upstream.sha,upstream.write
MODULE='scripts.flow_matching.audit_common105_test_label_oracle_v1'
CONFIG='configs/reproducibility/fm_common105_all577_test_label_oracle.v1.json'
TEST='tests/test_fm_common105_test_label_oracle_v1.py'
REPORT='full_grouped_test_label_oracle_diagnostic.json'
PROFILE='configs/reproducibility/fm_crossad_full139_independent_metrics.v1.json'

def registration():
    old=read(ROOT/upstream.CONFIG);upstream.verify(old)
    profile=read(ROOT/PROFILE)['settings']
    c=copy.deepcopy(old);c.pop('diagnostic_protocol')
    pins={p['path'] for p in c['source_receipts']}
    pins.update([upstream.CONFIG,upstream.MODULE.replace('.','/')+'.py',exact.CONFIG,
        exact.MODULE.replace('.','/')+'.py',PROFILE,MODULE.replace('.','/')+'.py',TEST])
    c.update(source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(pins)],
        state_root='experiments/runs/fm_common105_all577_test_label_oracle_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/common105_all577_test_label_oracle_v1',
        settings=dict(minimum_available_memory_bytes=16*2**30,
            minimum_commit_headroom_bytes=profile['minimum_commit_headroom_bytes'],
            emergency_commit_headroom_bytes=profile['emergency_commit_headroom_bytes'],
            maximum_worker_rss_bytes=profile['maximum_rss_bytes'],maximum_worker_private_bytes=profile['maximum_private_bytes']),
        resource_profile_source=PROFILE,inter_case_yield_seconds=20,
        posthoc_test_label_selected_diagnostic=True,prospectively_registered_benchmark=False,
        test_labels_used_to_select_oracle_threshold=True,test_labels_used_for_PA=True,
        fixed_original_validation_controls_test_label_threshold_selection=False,
        oracle_scores_never_enter_strict_leaderboards=True,model_training_or_reinference_performed=False,
        entity_boundaries_preserved_before_PA=True,threshold_scope='one global threshold across all original entities per label view',
        threshold_search='all exact score tie groups,integer F1 cross-products,strict greater,highest-cutoff tie break',
        boundary='New explicit posthoc test-label-selected diagnostic for all577matched source cases/1726label views. '
            'Require full successful source+immutable full scalar/label proof. All original score arrays/data/labels/points retained. '
            'Original strict validation metrics copied unchanged;exact raw and PA upper bounds are diagnostic only. '
            'PA is computed independently inside each original entity before global tie sweep;no merged boundary events. '
            'Resource profile inherited from existing bounded pure-score CrossAD audit,not a change to original source/native/TAB gates. '
            'No model fit/reinference/retry/new training-seed credit;yield original audit lock20seconds between complete cases. '
            'All873matched cases/2037views and45/681/273/702original experiments/native/fullTAB obligations remain.')
    return c

def verify(c):
    if c!=registration():raise ValueError('All577cases/1726views and explicitly test-label-selected diagnostic contract required')
    return c['cases']

def grouped_oracle(entities):
    """One global cutoff, while true anomaly events never cross entities."""
    if not entities:raise ValueError('Every full original entity required')
    scores=[];labels=[];adjusted=[];parts=[]
    for name in sorted(entities):
        y,x=entities[name];y=list(y);x=list(map(float,x))
        if any(v not in (0,1) for v in y):raise ValueError('Original binary labels required')
        y=list(map(int,y))
        if not len(y) or len(y)!=len(x):raise ValueError('Every aligned full entity required')
        parts.append((name,y,x));scores.extend(x);labels.extend(y);adjusted.extend(exact.pa_scores(x,y))
    raw=exact.exact_best_f1(scores,labels);pa=exact.exact_best_f1(adjusted,labels)
    for result,uses_pa in [(raw,False),(pa,True)]:
        counts=dict(true_positives=0,false_positives=0,false_negatives=0)
        for name,y,x in parts:
            predicted=[v>result['threshold'] for v in x]
            if uses_pa:predicted=exact.source.point_adjust(y,predicted)
            observed=exact.source.confusion(y,predicted)
            for k in counts:counts[k]+=observed[k]
        if any(result[k]!=counts[k] for k in counts):raise ValueError('Full grouped predictions must independently reproduce oracle sweep counts')
        result.update(point_adjustment=uses_pa,test_labels_used_for_event_postprocessing=uses_pa,
            independent_entity_groups=len(entities),entity_boundaries_preserved=True)
    return dict(raw=raw,PA=pa,global_threshold_per_label_view=True,per_entity_threshold_optimization=False,
        all_test_points=len(scores),test_anomaly_points=sum(labels))

def worker(c,case):
    import numpy as np
    basis=upstream.ready_case(case)
    if basis is None or basis['kind']!='source_complete':raise ValueError('Full actual source and scalar proof required')
    queue,job,result,data,binding=upstream.upstream.source_binding(c,case)
    scalar=read(basis['upstream_report'])
    if scalar['source_binding']!=binding:raise ValueError('Immutable full source binding required')
    output=Path(c['artifact_root'])/case['id'];output.mkdir(parents=True,exist_ok=False)
    views={}
    with np.load(ROOT/data['input_npz'],allow_pickle=False) as inputs,np.load(Path(binding['result_path']).parent/'scores.npz',allow_pickle=False) as saved:
        _,_,_,coverage,digests=upstream.upstream.original.collect_arrays(inputs,saved,data)
        if coverage!=scalar['full_positional_saved_score_coverage'] or digests!=scalar['source_array_digests']:
            raise ValueError('All complete original arrays/points must match previous full scalar audit')
        entries=sorted(data['entities'],key=lambda e:e['entity_id'])
        for policy,overlay in data['labels_only_controls'].items():
            if sha(overlay['path'])!=overlay['sha256']:raise ValueError('Frozen raw label overlay changed')
            with np.load(overlay['path'],allow_pickle=False) as labels:
                if set(labels.files)!={e['array_prefix']+'_labels' for e in entries}:raise ValueError('Every full original entity label required')
                entities={e['entity_id']:(labels[e['array_prefix']+'_labels'],saved[e['array_prefix']+'_test']) for e in entries}
                value=grouped_oracle(entities)
            previous=scalar['label_view_results'][policy]
            value.update(original_validation_1percent_noPA=previous['metrics']['strict']['1']['micro'],
                original_same_validation_threshold_entitywise_PA=previous['primary_protocol_diagnostics']['same_validation_threshold_entitywise_PA_diagnostic'],
                original_fixed_control_test_labels_used_to_select_threshold=False,
                oracle_test_labels_used_to_select_threshold=True,label_policy_selected_by_TEST_performance=False,
                between_training_seed_uncertainty=None)
            views[policy]=value
    if upstream.upstream.source_binding(c,case)[-1]!=binding or upstream.ready_case(case)!=basis:
        raise ValueError('Original source/score/checkpoint/scalar lineage changed during diagnostic')
    write(output/REPORT,dict(passed=True,case=case,configuration_sha256=sha(ROOT/CONFIG),source_binding=binding,
        upstream_scalar_proof=basis,source_array_digests=digests,full_positional_saved_score_coverage=coverage,
        label_view_results=views,required_label_view_results=len(views),
        all_original_point_populations_retained=True,entity_boundaries_preserved_before_PA=True,
        test_labels_used_to_select_oracle_threshold=True,posthoc_test_label_selected_diagnostic=True,
        oracle_scores_never_enter_strict_leaderboards=True,original_validation_controls_unchanged=True,
        model_training_or_reinference_performed=False,independent_model_reinference_certified=False,
        full_TAB_or_author_equivalence_certified=False,all_original_experiments_complete=False))

def controller(c,cases):
    previous=upstream.upstream.original
    node=next(n for n in ast.parse(Path(previous.__file__).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='controller')
    class Bind(ast.NodeTransformer):
        def visit_Constant(self,n):
            if type(n.value) is int and n.value in (873,2037):return ast.copy_location(ast.Constant({873:577,2037:1726}[n.value]),n)
            if isinstance(n.value,str):return ast.copy_location(ast.Constant(n.value.replace('873','577').replace('2037','1726')),n)
            return n
        def visit_Expr(self,n):
            self.generic_visit(n)
            if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='update' and any(k.arg=='state' and isinstance(k.value,ast.Constant) and k.value.value=='between_complete_label_metric_audits' for k in n.value.keywords):
                sleep=ast.parse("time.sleep(config['inter_case_yield_seconds'])").body[0]
                return [n,sleep]
            return n
    ns=dict(previous.controller.__globals__,CONFIG=CONFIG,MODULE=MODULE,REPORT=REPORT,verify=verify,
        source_status=upstream.ready_case,source_binding=upstream.upstream.source_binding)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Bind().visit(node)],type_ignores=[])),str(previous.__file__),'exec'),ns)
    return ns['controller'](c,cases)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--job-id');a=p.parse_args()
    if a.register:
        from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive
        exclusive(ROOT/CONFIG,registration());return
    c=read(ROOT/CONFIG);cases=verify(c)
    if a.job_id:worker(c,next(v for v in cases if v['id']==a.job_id))
    else:controller(c,cases)

if __name__=='__main__':main()
