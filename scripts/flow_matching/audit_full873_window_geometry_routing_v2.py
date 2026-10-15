"""All873 actual array headers, distinguishing the real MOMENT worker route.

V1 assumed every family used the generic unpadded window builder. MOMENT uses
its own segment_windows with explicit edge replication. Preserve V1, correct
its six false-positive job predictions, and disclose the observed-fill policy.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
from scripts.flow_matching import audit_full873_window_geometry_v1 as prior

scalar=prior.scalar
ROOT,read,sha,write=scalar.ROOT,scalar.read,scalar.sha,scalar.write


def actual_moment_route(execution):
    family=next(f for f in execution['families'] if f['name']=='moment')
    controller=ROOT/family['controller_source'];worker=ROOT/'scripts/flow_matching/run_moment_pretrained_job.py'
    if family['controller_source']!='scripts/flow_matching/run_moment_pretrained_queue.py':
        raise ValueError('The actual specialized MOMENT controller must be checked')
    tree=ast.parse(controller.read_text(encoding='utf-8'))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and isinstance(n.func.value,ast.Name) and n.func.value.id=='subprocess' and n.func.attr=='Popen']
    if len(calls)!=1 or not any(isinstance(n,ast.Constant) and n.value=='scripts/flow_matching/run_moment_pretrained_job.py' for n in ast.walk(calls[0])):
        raise ValueError('Actual controller must launch the specialized MOMENT worker')
    tree=ast.parse(worker.read_text(encoding='utf-8'))
    definitions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    segment=ast.unparse(definitions['segment_windows']);scores=ast.unparse(definitions['segment_scores'])
    if ("mode='edge'" not in segment or 'window - len(values)' not in segment
            or 'raw[0, :len(values)]' not in scores or "'all_context_tokens_observed': True" not in scores):
        raise ValueError('Pinned observed edge-fill/crop policy must remain explicit')
    if not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='segment_scores' for n in ast.walk(definitions['run'])):
        raise ValueError('Specialized scoring function must actually be called')
    return dict(controller=str(controller),controller_sha256=sha(controller),worker=str(worker),worker_sha256=sha(worker),
        short_series_storage='right_edge_replication',fill_tokens_observed=True,fill_targets_scored=False,
        fill_changes_real_context=True,generic_short_sequence_rejection_applies=False)


def classify(row,moment_route):
    value=dict(row)
    short=row['incompatible_role_inputs']
    if row['case']['family']=='moment':
        if any(r['points']<1 for r in short):raise ValueError('Specialized worker still needs nonempty sources')
        value.update(actual_numerical_route=moment_route,short_roles_handled_by_worker=short,
            actual_worker_predicted_incompatible_inputs=[],actual_worker_predicted_incompatible=False,
            predicted_rejection_before_fit=False,predicted_rejection_after_fit=False,
            correction_to_v1=bool(short))
    else:
        value.update(actual_numerical_route='original_run_tsad_experiment_unpadded_covering_windows',
            short_roles_handled_by_worker=[],actual_worker_predicted_incompatible_inputs=short,
            actual_worker_predicted_incompatible=bool(short),
            predicted_rejection_before_fit=row['training_rejected_before_fit'],
            predicted_rejection_after_fit=row['possible_later_scoring_rejection_after_fit'],correction_to_v1=False)
    value['actual_job_terminal_failure_observed']=False
    return value


def audit():
    old=prior.audit();config=read(ROOT/old['source_configuration'])
    execution=read(ROOT/config['source_execution']);route=actual_moment_route(execution)
    rows=[classify(r,route) for r in old['records']]
    failures=[r for r in rows if r['actual_worker_predicted_incompatible']]
    corrections=[r for r in rows if r['correction_to_v1']]
    if len(rows)!=873 or len(failures)!=5 or len(corrections)!=6 or {r['case']['family'] for r in failures}!={'baselines'}:
        raise ValueError('All873actual routes must explain exactly5DCD conflicts and6MOMENT V1 false positives')
    return dict(passed=True,source_configuration=old['source_configuration'],source_configuration_sha256=old['source_configuration_sha256'],
        manifest_path=old['manifest_path'],manifest_sha256=old['manifest_sha256'],all_source_cases=873,
        actual_arrays_headers_checked=old['actual_arrays_headers_checked'],
        raw_short_window_case_count=11,actual_worker_predicted_incompatible_cases=5,
        corrected_MOMENT_false_positive_cases=6,original_MOMENT_cases_unchanged=18,
        incompatible_case_counts_by_family=dict(Counter(r['case']['family'] for r in failures)),
        predicted_rejection_before_fit=sum(r['predicted_rejection_before_fit'] for r in rows),
        predicted_rejection_after_fit=sum(r['predicted_rejection_after_fit'] for r in rows),
        actual_terminal_failures_certified=0,records=rows,
        historical_v1_report_preserved=True,existing_jobs_changed_cancelled_or_retried=False,
        masked_MOMENT_is_additional_preprocessing_control_not_required_crash_fix=True,
        full_original_scope=old['full_original_scope'],all_original_experiments_complete=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    path=Path(a.output)
    if path.exists():raise FileExistsError('Preserve original routing observation')
    report=audit();write(path,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['records','full_original_scope']}))


if __name__=='__main__':main()
