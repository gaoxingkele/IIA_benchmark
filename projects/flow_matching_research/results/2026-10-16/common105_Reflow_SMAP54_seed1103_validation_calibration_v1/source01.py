"""Post-hoc fixed-score calibration diagnostic for all577 matched full-entity cases.

The two fixed policies use validation scores only. All source label views are
retained. This never replaces original metrics, selects a winning test policy,
fits a model or certifies pending native model reinference/TAB results.
"""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
from types import FunctionType
from scripts.flow_matching import audit_common105_matched577_labels_v1 as upstream

ROOT,read,sha,write=upstream.ROOT,upstream.read,upstream.sha,upstream.write
MODULE='scripts.flow_matching.audit_common105_validation_calibration_v1'
CONFIG='configs/reproducibility/fm_common105_all577_validation_calibration.2026-10-16_v1.json'
TEST='tests/test_fm_common105_validation_calibration_v1.py'
REPORT='fixed_validation_calibration_diagnostic.json'


def registration():
    old=read(ROOT/upstream.CONFIG);upstream.verify(old)
    paths={r['path'] for r in old['source_receipts']}
    paths.update([upstream.CONFIG,MODULE.replace('.','/')+'.py',TEST,
        'src/iia_benchmark/evaluation/mtsad_strict.py'])
    return dict(old,upstream_label_configuration=upstream.CONFIG,
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(paths)],
        state_root='experiments/runs/fm_common105_all577_validation_calibration_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/common105_all577_validation_calibration_v1',
        diagnostic_protocol=dict(global_and_entity_validation_false_alarm_percent=1,
            percentile_method='linear',prediction_operator='strict_greater',
            paired_entity_bootstrap_resamples=200,bootstrap_seed='original_model_seed',
            thresholds_never_fit_on_test_scores_or_any_test_labels=True,
            all_label_views_share_the_same_two_frozen_prediction_arrays=True,
            new_posthoc_diagnostic_not_prospective_preregistration=True),
        boundary='Post-hoc calibration diagnostic on every577 matched full-entity source model and1726raw label views. '
            'Successful full source and immutable original scalar/label proof required. '
            'Compare original pooled validation99th percentile with each entity own validation99th percentile. '
            'Fixed existing1percent alarm budget,all original entities/points/labels,no test label or test-score threshold fit. '
            'Paired200entity resamples with source training seed;this is fixed-model entity uncertainty,not training-seed CI. '
            'Original source metrics/predictions,all resource policies,original native/TAB queues and full45/681/273/702scope preserved. '
            'No PA,no new model fit/reinference,no label/view winner selected,no fullTAB or author certification.')


def verify(config):
    if config!=registration():raise ValueError('All577cases/1726views and both fixed validation-only policies required')
    return config['cases']


def ready_case(case):
    state=upstream.original.source_status(case)
    if state is None or state['kind']=='source_failed_preserved':return state
    config=read(ROOT/upstream.CONFIG)
    report=Path(config['artifact_root'])/case['id']/upstream.REPORT
    guard=ROOT/config['state_root']/'attempts'/case['id']/'guard_receipt.json'
    if not report.exists() or not guard.exists():return None
    proof,g=read(report),read(guard)
    if (not g['passed'] or g['exit_code']!=0 or g['resource_guard_aborted'] or not proof['passed']
            or g['proof_sha256']!=sha(report) or proof['case']!=case
            or proof['configuration_sha256']!=sha(ROOT/upstream.CONFIG)
            or not proof['source_metrics_exactly_recomputed'] or not proof['all_original_arrays_checked']
            or not proof['all_validation_thresholds_and_predictions_unchanged']
            or set(proof['label_view_results'])!=set(case['required_label_policies'])):
        raise ValueError('Actual complete original full-score/label scalar proof and successful guard required')
    return dict(state,upstream_report=str(report),upstream_report_sha256=sha(report),upstream_guard_sha256=sha(guard))


def validation_thresholds(validation):
    """No test scores or labels are accepted by this calibration function."""
    import numpy as np
    if not validation:raise ValueError('Every original entity requires validation scores')
    arrays={}
    for name in sorted(validation):
        a=np.asarray(validation[name],dtype=np.float64)
        if a.ndim!=1 or not len(a) or not np.isfinite(a).all():raise ValueError('Full finite one-dimensional validation scores required')
        arrays[name]=a
    return dict(global_threshold=float(np.percentile(np.concatenate(list(arrays.values())),99,method='linear')),
        per_entity_thresholds={k:float(np.percentile(v,99,method='linear')) for k,v in arrays.items()},
        false_alarm_percent=1,fit_split='validation',percentile_method='linear')


def confusion(labels,prediction):
    import numpy as np
    y,p=np.asarray(labels),np.asarray(prediction)
    if y.ndim!=1 or y.shape!=p.shape or not len(y) or not np.isin(y,[0,1]).all() or p.dtype!=np.bool_:
        raise ValueError('Every binary target and frozen boolean prediction must align')
    y=y.astype(bool)
    return [int(np.count_nonzero(y&p)),int(np.count_nonzero(~y&p)),
        int(np.count_nonzero(y&~p)),int(np.count_nonzero(~y&~p))]


def point_metrics(counts):
    import numpy as np
    a=np.asarray(counts)
    if a.shape!=(4,) or not np.issubdtype(a.dtype,np.integer) or (a<0).any():raise ValueError('Full nonnegative integer TP/FP/FN/TN counts required')
    tp,fp,fn,tn=map(int,a)
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
    # Retain source floating arithmetic; the algebraically equivalent direct
    # count formula can differ by one ulp for some labels-only controls.
    return dict(precision=precision,recall=recall,
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.,TP=tp,FP=fp,FN=fn,TN=tn)


def paired_bootstrap(global_counts,entity_counts,seed):
    import numpy as np
    a,b=np.asarray(global_counts,dtype=np.int64),np.asarray(entity_counts,dtype=np.int64)
    if (a.ndim!=2 or a.shape[1]!=4 or a.shape!=b.shape or not len(a) or (a<0).any() or (b<0).any()
            or not np.array_equal(a.sum(axis=1),b.sum(axis=1))
            or not np.array_equal(a[:,0]+a[:,2],b[:,0]+b[:,2])):
        raise ValueError('Paired predictions on every same full entity and label population required')
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(200):
        indices=rng.integers(0,len(a),len(a))
        samples.append(point_metrics(b[indices].sum(axis=0))['f1']-point_metrics(a[indices].sum(axis=0))['f1'])
    ci=np.percentile(samples,[2.5,97.5],method='linear')
    return dict(seed=seed,resamples=200,independent_entity_groups=len(a),
        observed_F1_difference=point_metrics(b.sum(axis=0))['f1']-point_metrics(a.sum(axis=0))['f1'],
        entity_bootstrap_F1_difference_ci95=ci.tolist(),paired_F1_difference_samples=samples,
        uncertainty_scope='conditional on this fixed trained model and full frozen split;not training seed uncertainty')


def worker(config,case):
    import numpy as np
    basis=ready_case(case)
    if basis is None or basis['kind']!='source_complete':raise ValueError('Completed full source/label scalar evidence required')
    queue,job,result,data,binding=upstream.source_binding(config,case)
    original=read(basis['upstream_report'])
    if original['source_binding']!=binding:raise ValueError('Original full source/data/checkpoint/score binding changed')
    if queue['evaluation']['primary_false_alarm_percent']!=1:raise ValueError('Keep source1percent primary alarm budget')
    output=Path(config['artifact_root'])/case['id']
    output.mkdir(parents=True,exist_ok=False)
    score_path=ROOT/job['output_directory']/'scores.npz'
    with np.load(ROOT/data['input_npz'],allow_pickle=False) as inputs,np.load(score_path,allow_pickle=False) as saved:
        _,_,_,coverage,digests=upstream.original.collect_arrays(inputs,saved,data)
        if coverage!=original['full_positional_saved_score_coverage'] or digests!=original['source_array_digests']:
            raise ValueError('Every actual original score/target coordinate required')
        entries=sorted(data['entities'],key=lambda e:e['entity_id'])
        validations={e['entity_id']:saved[e['array_prefix']+'_validation'] for e in entries}
        thresholds=validation_thresholds(validations)
        if thresholds['global_threshold']!=result['metrics']['strict']['1']['calibration']['threshold']:
            raise ValueError('Original pooled validation threshold must reproduce exactly')
        predictions={};pred_payload={};calibration_rows=[]
        for e in entries:
            name,prefix=e['entity_id'],e['array_prefix'];scores=saved[prefix+'_test'];v=validations[name]
            global_prediction=scores>thresholds['global_threshold']
            entity_prediction=scores>thresholds['per_entity_thresholds'][name]
            predictions[name]=(global_prediction,entity_prediction)
            pred_payload[prefix+'_global_prediction']=global_prediction
            pred_payload[prefix+'_entity_prediction']=entity_prediction
            calibration_rows.append(dict(entity=name,validation_points=len(v),test_points=len(scores),
                global_threshold=thresholds['global_threshold'],entity_threshold=thresholds['per_entity_thresholds'][name],
                global_validation_alarm_fraction=float(np.mean(v>thresholds['global_threshold'])),
                entity_validation_alarm_fraction=float(np.mean(v>thresholds['per_entity_thresholds'][name]))))
        views={}
        for policy,label in data['labels_only_controls'].items():
            if sha(label['path'])!=label['sha256']:raise ValueError('Frozen source label overlay changed')
            rows=[];a=[];b=[]
            with np.load(label['path'],allow_pickle=False) as labels:
                if set(labels.files)!={e['array_prefix']+'_labels' for e in entries}:raise ValueError('Every original entity label required')
                for e in entries:
                    name,prefix=e['entity_id'],e['array_prefix'];y=labels[prefix+'_labels'];pg,pe=predictions[name]
                    cg,ce=confusion(y,pg),confusion(y,pe);a.append(cg);b.append(ce)
                    rows.append(dict(entity=name,global_policy=point_metrics(cg),entity_policy=point_metrics(ce)))
            global_metrics=point_metrics(np.asarray(a).sum(axis=0));entity_metrics=point_metrics(np.asarray(b).sum(axis=0))
            previous=original['label_view_results'][policy]['metrics']['strict']['1']['micro']
            if any(global_metrics[k]!=previous[k] for k in ['precision','recall','f1']):raise ValueError('Original global primary scores must remain exact')
            views[policy]=dict(global_policy=global_metrics,entity_policy=entity_metrics,
                paired_entity_uncertainty=paired_bootstrap(a,b,case['seed']),per_entity=rows,
                label_policy_selected_by_TEST_performance=False,point_adjustment=False)
    prediction_path=output/'both_fixed_prediction_arrays.npz'
    with prediction_path.open('xb') as stream:np.savez_compressed(stream,**pred_payload)
    if upstream.source_binding(config,case)[-1]!=binding or ready_case(case)!=basis:
        raise ValueError('Source data/checkpoints/scores/labels/scalar proof changed during readonly diagnostic')
    proof=dict(passed=True,case=case,configuration_sha256=sha(ROOT/CONFIG),source_binding=binding,
        upstream_scalar_proof=basis,thresholds=thresholds,calibration_by_entity=calibration_rows,
        label_view_results=views,source_score_digests=digests,complete_source_point_coverage=coverage,
        predictions_path=str(prediction_path),predictions_sha256=sha(prediction_path),
        original_primary_global_metrics_reproduced_exactly=True,all_entities_retained=True,
        new_posthoc_diagnostic_not_prospective_preregistration=True,any_test_label_or_score_used_to_fit_threshold=False,
        model_training_or_reinference_performed=False,independent_model_reinference_certified_by_this_diagnostic=False,
        full_TAB_or_historical_author_equivalence_certified=False,original_scope=config['original_scope'],all_original_experiments_complete=False)
    write(output/REPORT,proof)


def controller(config,cases):
    """Reuse the guarded scalar controller; alter scope/binding, never numeric caps."""
    previous=upstream.original
    node=next(n for n in ast.parse(Path(previous.__file__).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='controller')
    class ScopeNames(ast.NodeTransformer):
        def visit_Constant(self,n):
            if type(n.value) is int and n.value in (873,2037):return ast.copy_location(ast.Constant({873:577,2037:1726}[n.value]),n)
            if isinstance(n.value,str):return ast.copy_location(ast.Constant(n.value.replace('873','577').replace('2037','1726')),n)
            return n
    ns=dict(previous.controller.__globals__,CONFIG=CONFIG,MODULE=MODULE,REPORT=REPORT,verify=verify,
        source_status=ready_case,source_binding=upstream.source_binding)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ScopeNames().visit(node)],type_ignores=[])),str(previous.__file__),'exec'),ns)
    return ns['controller'](config,cases)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true');parser.add_argument('--job-id')
    args=parser.parse_args()
    if args.register:
        from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive
        exclusive(ROOT/CONFIG,registration());return
    config=read(ROOT/CONFIG);cases=verify(config)
    if args.job_id:worker(config,next(c for c in cases if c['id']==args.job_id))
    else:controller(config,cases)


if __name__=='__main__':main()
