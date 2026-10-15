"""Complete five-seed statistics after all500 full checkpoint reviews.

100 dataset cohorts,95 paired comparisons on identical test prefixes,and20
five-dataset macro cohorts. No training,model inference,retry or leaderboard
certification. Every producer receipt and original frozen coverage is retained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from scripts.flow_matching import run_anomaly_transformer_full_checkpoint_review_v1 as producer

ROOT, read, sha, write = producer.ROOT, producer.read, producer.sha, producer.write
PRODUCER = 'configs/runtime/fm_anomaly_transformer_full_checkpoint_review.v1.json'
MODULE = 'scripts.flow_matching.run_anomaly_transformer_full_cohort_statistics_v1'
SEEDS = [1103, 1104, 1105, 1106, 1107]
DATASETS = ['SMD', 'MSL', 'SMAP', 'PSM', 'SWaT']
DF4_CRITICAL_975 = 2.7764451051977987


def variant(case):
    family, job = case['family'], case['job']
    if family == 'sensitivity':
        return 'sensitivity_'+job['axis']+str(job['value'])
    if family == 'max':
        return job['variant']
    return family


def primary_stream(case):
    return 'all.reconstruction' if case['family'] == 'plain' else 'all.multiplication'


def tasks(cases):
    groups = {}
    for identity, case in cases.items():
        key = variant(case)+'__'+case['job']['dataset'].lower()
        group = groups.setdefault(key, dict(kind='cohort', id=key, variant=variant(case),
            dataset=case['job']['dataset'], cases=[]))
        group['cases'].append(identity)
    for group in groups.values():
        group['cases'].sort(key=lambda identity: cases[identity]['job']['seed'])
        if [cases[i]['job']['seed'] for i in group['cases']] != SEEDS:
            raise ValueError('Exactly five unique original seeds required per dataset cohort')
    if len(groups) != 100 or len({i for g in groups.values() for i in g['cases']}) != 500:
        raise ValueError('All100complete dataset cohorts must retain all500unique models')
    result = dict(groups)
    for key, group in groups.items():
        if group['variant'] != 'main':
            identity = 'paired__'+key
            baseline = 'main__'+group['dataset'].lower()
            result[identity] = dict(kind='paired', id=identity, variant=group['variant'],
                dataset=group['dataset'], dependencies=[key, baseline],
                cases=group['cases']+groups[baseline]['cases'])
    for name in dict.fromkeys(g['variant'] for g in groups.values()):
        dependencies = [name+'__'+d.lower() for d in DATASETS]
        identity = 'macro__'+name
        result[identity] = dict(kind='macro', id=identity, variant=name, dependencies=dependencies,
            cases=[i for key in dependencies for i in groups[key]['cases']])
    if len(result) != 215:
        raise ValueError('All100cohorts,95paired comparisons and20macro summaries required')
    return result


def verify(runtime):
    source = read(ROOT/PRODUCER)
    if (runtime['producer'] != PRODUCER or runtime['producer_sha256'] != sha(ROOT/PRODUCER)
            or runtime['settings'] != source['settings'] or runtime['resource_lock'] != source['resource_lock']
            or runtime['python'] != source['python'] or runtime['original_scope'] != source['original_scope']
            or runtime['seeds'] != SEEDS or runtime['datasets'] != DATASETS
            or runtime['student_t_critical_df4_975'] != DF4_CRITICAL_975
            or runtime['confidence_level'] != .95 or runtime['required_models'] != 500
            or runtime['required_tasks'] != 215 or runtime['paired_test_support'] != 'common_frozen_covered_prefix'
            or runtime['paired_calibration'] != 'each_model_original_full_validation_scores'
            or runtime['macro_weighting'] != 'equal_dataset_weight_within_each_seed'
            or runtime['primary_endpoint'] != 'validation_ratio.no_PA.f1'
            or runtime['fixed_delta_controls_are_diagnostic'] is not True
            or runtime['interval_scope'] != 'descriptive_training_seed_variability_only_unclipped'):
        raise ValueError('Complete original cohorts and frozen statistics/resource policy required')
    for pin in runtime['source_receipts']:
        if sha(ROOT/pin['path']) != pin['sha256']:
            raise ValueError('Frozen complete cohort input changed: '+pin['path'])
    cases, contract = producer.verify(source)
    selected = tasks(cases)
    if runtime['tasks'] != list(selected.values()):
        raise ValueError('All frozen source-to-cohort/task bindings required')
    return source, cases, contract, selected


def five_seed_summary(values, seeds=SEEDS):
    if seeds != SEEDS or len(values) != 5 or any(isinstance(v, bool) or not math.isfinite(v) for v in values):
        raise ValueError('All five finite original seed observations required; no partial CI')
    values = [float(v) for v in values]
    mean, std = statistics.mean(values), statistics.stdev(values)
    se = std/math.sqrt(5); half = DF4_CRITICAL_975*se
    return dict(seeds=list(seeds), values=values, n=5, mean=mean, sample_standard_deviation=std,
        standard_error=se, student_t_df4_95_interval=[mean-half, mean+half],
        interval_is_unclipped=True, median=statistics.median(values), minimum=min(values), maximum=max(values),
        leave_one_seed_out_means={str(s):statistics.mean(values[:i]+values[i+1:]) for i,s in enumerate(seeds)},
        interval_scope='training_seed_variability_only', multiple_comparison_correction_applied=False,
        hypothesis_test_or_generalization_guarantee=False)


def metric_leaves(tree, path=()):
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from metric_leaves(value, path+(key,))
    elif isinstance(tree, list):
        for i, value in enumerate(tree):
            yield from metric_leaves(value, path+(i,))
    elif tree is None or (isinstance(tree, (int, float)) and not isinstance(tree, bool)):
        yield path, tree


def aggregate_metric_trees(trees):
    leaves = [dict(metric_leaves(tree)) for tree in trees]
    if len(leaves) != 5 or any(set(row) != set(leaves[0]) for row in leaves):
        raise ValueError('Every original metric stream/control must exist in all five seeds')
    result = []
    for path in leaves[0]:
        values = [row[path] for row in leaves]
        if all(v is None for v in values):
            record = dict(defined_seeds=0, summary=None, reason='undefined_in_all_five_original_results')
        elif any(v is None for v in values):
            raise ValueError('Mixed undefined seed metrics must not silently drop observations')
        else:
            record = dict(defined_seeds=5, summary=five_seed_summary(values))
        result.append(dict(metric_path=list(path), **record))
    return result


def primary_metrics(evaluation):
    return {**{protocol+'.'+adjustment+'.'+metric:evaluation[protocol][adjustment][metric]
               for protocol in ['validation_ratio', 'fixed_delta']
               for adjustment in ['no_PA', 'paper_PA'] for metric in ['precision', 'recall', 'f1']},
            'continuous_unadjusted_score_AUROC': evaluation['continuous_unadjusted_score_AUROC'],
            'continuous_score_AP': evaluation['continuous_score_AP']}


def stage_paths(runtime, identity):
    return (Path(runtime['artifact_root'])/identity/'statistics.json',
            ROOT/runtime['state_root']/'attempts'/identity/'guard_receipt.json')


def read_completed_task(runtime, task):
    output, receipt = stage_paths(runtime, task['id'])
    report, guard = read(output), read(receipt)
    if (not report['passed'] or report['task'] != task or guard['task'] != task['id']
            or not guard['passed'] or guard['exit_code'] != 0 or guard['resource_guard_aborted']
            or guard['proof_sha256'] != sha(output)
            or report['runtime_sha256'] != sha(ROOT/runtime['_config_path'])):
        raise ValueError('Unchanged complete guarded cohort/statistics task required')
    return report


def producer_proof_paths(source, identity):
    base = Path(source['artifact_root'])/identity
    state = ROOT/source['state_root']/'attempts'/identity
    return [(base/'native/native_replay.json', state/'native/guard_receipt.json'),
            (base/'independent/review.json', state/'independent/guard_receipt.json')]


def read_full_model(source, case, contract):
    result, audit, binding = producer.check_source(case, contract)
    paths = producer_proof_paths(source, case['job']['id']); proofs = []
    for phase, (path, receipt) in zip(['native', 'independent'], paths):
        proof, guard = read(path), read(receipt)
        command = [source['python'], '-B', '-X', 'utf8', '-u', '-m', producer.MODULE,
            '--config', PRODUCER, '--stage', phase, '--job-id', case['job']['id']]
        if (not proof['passed'] or proof['job'] != case['job'] or proof['source_binding'] != binding
                or not guard['passed'] or guard['exit_code'] != 0 or guard['resource_guard_aborted']
                or guard['proof_sha256'] != sha(path) or guard['exact_child_handle']['command'] != command
                or guard['source_configuration'] != case['path']
                or guard['source_configuration_sha256'] != binding['configuration_sha256']):
            raise ValueError('Every original full native and independent proof/guard must pass')
        proofs.append(proof)
    native, independent = proofs
    if (native['runtime_sha256'] != sha(ROOT/PRODUCER)
            or not native['all_covered_points_reinferred'] or not native['native_dtype_shape_content_identical']
            or native['training_updates_performed'] != 0
            or independent['native_replay_sha256'] != sha(paths[0][0])
            or not independent['all_original_streams_and_covered_arrays_checked']
            or not independent['all_original_thresholds_confusion_counts_PA_and_ranking_metrics_checked']
            or independent['test_used_for_calibration'] or independent['strict_TAB_result']):
        raise ValueError('Full original verification boundary differs')
    for row in native['records']:
        if sha(row['replay_path']) != row['replay_sha256']:
            raise ValueError('Full native score artifact changed after independent verification')
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import require_same_metrics
    require_same_metrics(result['metrics'], independent['independent_metrics'])
    evidence = dict(id=case['job']['id'], seed=case['job']['seed'], source_binding=binding,
        native_proof_sha256=sha(paths[0][0]), independent_proof_sha256=sha(paths[1][0]),
        native_guard_sha256=sha(paths[0][1]), independent_guard_sha256=sha(paths[1][1]),
        data_manifest_sha256=result['data_manifest_sha256'], covered_points=result['covered_points'],
        omitted_tail_points=result['omitted_tail_points'], primary_stream=primary_stream(case))
    return result, independent['independent_metrics'], evidence


def paired_common_support(rows, baseline_rows, cases, baseline_cases, contract):
    """Preserve full validation calibration; independently score identical truth prefixes."""
    import numpy as np
    from scripts.flow_matching import anomaly_transformer_independent_math_v1 as independent
    dataset = cases[0]['job']['dataset']; evaluation = contract['paper_evaluation']
    differences = []; evidence = []
    for row, baseline, case, basecase in zip(rows, baseline_rows, cases, baseline_cases):
        if case['job']['seed'] != basecase['job']['seed'] or case['job']['dataset'] != basecase['job']['dataset']:
            raise ValueError('Paired comparison requires the same original seed and dataset')
        a, acache, _ = producer.source_paths(case); b, bcache, _ = producer.source_paths(basecase)
        count = min(row['covered_points']['test'], baseline['covered_points']['test'])
        truth = np.asarray(np.load(acache/'labels.npy', mmap_mode='r')[:count])
        independent.require_identical(truth, np.asarray(np.load(bcache/'labels.npy', mmap_mode='r')[:count]))
        recalculated = []
        for base, current, result in [(a, case, row), (b, basecase, baseline)]:
            stream = primary_stream(current)
            with np.load(base/'test_scores.npz', allow_pickle=False) as test, np.load(base/'validation_scores.npz', allow_pickle=False) as validation:
                metric = independent.independent_evaluation(truth, test[stream][:count], validation[stream],
                    evaluation['r_percent'][dataset], evaluation['fixed_delta'][dataset],
                    evaluation['ROC_predefined_r_percent'], current['family'] == 'plain')
            if metric['validation_ratio']['threshold'] != result['metrics'][stream]['validation_ratio']['threshold']:
                raise ValueError('Common test support must not alter original full-validation calibration')
            if count == result['covered_points']['test']:
                independent.require_same_metrics(result['metrics'][stream], metric)
            recalculated.append(metric)
        left, right = map(primary_metrics, recalculated)
        if set(left) != set(right) or any(left[k] is None or right[k] is None for k in left):
            raise ValueError('All five paired performance observations must be defined')
        differences.append({k:left[k]-right[k] for k in left})
        evidence.append(dict(seed=case['job']['seed'], common_test_points=count,
            variant_original_test_points=row['covered_points']['test'], baseline_original_test_points=baseline['covered_points']['test'],
            variant_primary_stream=primary_stream(case), baseline_primary_stream=primary_stream(basecase),
            variant_metrics=recalculated[0], baseline_metrics=recalculated[1],
            original_full_validation_thresholds_preserved=True))
    return {k:five_seed_summary([row[k] for row in differences]) for k in differences[0]}, evidence


def macro_summary(group_reports, case_by_id):
    if (len(group_reports) != 5 or [r['dataset'] for r in group_reports] != DATASETS
            or len({r['variant'] for r in group_reports}) != 1):
        raise ValueError('All five original datasets with equal weighting required; no partial macro')
    streams = set(group_reports[0]['per_seed_metrics'][0]['metrics'])
    if any(set(row['metrics']) != streams for report in group_reports for row in report['per_seed_metrics']):
        raise ValueError('All original model score criteria must be present in every dataset')
    by_seed = []
    for seed in SEEDS:
        metrics = {}
        for stream in sorted(streams):
            values = []
            for report in group_reports:
                entries = [row for row in report['per_seed_metrics'] if row['seed'] == seed]
                if len(entries) != 1 or case_by_id[entries[0]['id']]['job']['seed'] != seed:
                    raise ValueError('Exactly one original model per dataset and matching seed required')
                values.append(primary_metrics(entries[0]['metrics'][stream]))
            metrics[stream] = {key:statistics.mean([row[key] for row in values]) for key in values[0]}
        by_seed.append(dict(seed=seed, metrics=metrics))
    return by_seed, aggregate_metric_trees([row['metrics'] for row in by_seed])


def worker(runtime, source, cases, contract, task, all_tasks):
    output, _ = stage_paths(runtime, task['id'])
    output.parent.mkdir(parents=True, exist_ok=False)
    model_evidence = []; metric_rows = []; results = []
    if task['kind'] in ['cohort', 'paired']:
        for identity in task['cases']:
            result, metrics, evidence = read_full_model(source, cases[identity], contract)
            model_evidence.append(evidence); results.append(result)
            metric_rows.append(dict(id=identity, seed=cases[identity]['job']['seed'], metrics=metrics))
    details = {}
    if task['kind'] == 'cohort':
        if (len({r['data_manifest_sha256'] for r in results}) != 1
                or len({json.dumps(r['covered_points'], sort_keys=True) for r in results}) != 1
                or len({json.dumps(producer.parameters(cases[i]), sort_keys=True) for i in task['cases']}) != 1):
            raise ValueError('Identical frozen data,hyperparameters and covered support across five seeds required')
        details = dict(dataset=task['dataset'], variant=task['variant'],
            per_seed_metrics=metric_rows, summaries=aggregate_metric_trees([r['metrics'] for r in metric_rows]),
            primary_stream=primary_stream(cases[task['cases'][0]]))
    elif task['kind'] == 'paired':
        for dependency in task['dependencies']:
            read_completed_task(runtime, all_tasks[dependency])
        comparisons, evidence = paired_common_support(results[:5], results[5:],
            [cases[i] for i in task['cases'][:5]], [cases[i] for i in task['cases'][5:]], contract)
        details = dict(variant=task['variant'], dataset=task['dataset'], paired_variant_minus_main=comparisons,
            common_support_per_seed=evidence, lower_or_higher_is_better='positive_performance_difference_favors_variant',
            source_metrics_before_common_prefix_recalculation=metric_rows,
            test_labels_or_scores_used_to_choose_support=False)
    else:
        reports = [read_completed_task(runtime, all_tasks[i]) for i in task['dependencies']]
        by_seed, summaries = macro_summary(reports, cases)
        details = dict(variant=task['variant'], datasets=DATASETS, per_seed_dataset_macro=by_seed,
            summaries=summaries, macro_weighting='equal_dataset_weight_within_each_seed')
    dependencies = []
    for identity in task.get('dependencies', []):
        path, receipt = stage_paths(runtime, identity)
        dependencies.append(dict(id=identity, report_sha256=sha(path), guard_sha256=sha(receipt)))
    write(output, dict(passed=True, task=task, runtime_sha256=sha(ROOT/runtime['_config_path']),
        model_evidence=model_evidence, dependency_evidence=dependencies, **details,
        seeds=SEEDS, confidence_level=.95, student_t_df=4, student_t_critical=DF4_CRITICAL_975,
        confidence_intervals_describe_training_seed_variability_only=True,
        no_multiple_comparison_or_dataset_generalization_guarantee=True,
        primary_endpoint='validation_ratio.no_PA.f1', fixed_delta_controls_are_diagnostic=True,
        test_selected_hyperparameters=False, training_updates_performed=0, model_inference_performed=False,
        independent_raw_preprocessing_value_reconstruction_performed=False,
        paper_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False))


def readiness(runtime, source, task, done, failed):
    for dependency in task.get('dependencies', []):
        if dependency in failed:
            raise ValueError('Preserved failed prerequisite statistics task: '+dependency)
        if dependency not in done:
            return False
    if task['kind'] == 'macro':
        return True
    source_status = ROOT/source['state_root']/'status.json'
    source_failures = read(source_status).get('failed_preserved', {}) if source_status.exists() else {}
    for identity in task['cases']:
        if identity in source_failures:
            raise ValueError('Preserved failed original producer task: '+identity)
        for path, receipt in producer_proof_paths(source, identity):
            if not receipt.exists():
                return False
            guard = read(receipt)
            if not guard.get('passed') or guard.get('resource_guard_aborted') or guard.get('exit_code') != 0:
                raise ValueError('Preserved failed original full-model verification: '+identity)
            if not path.exists():
                raise ValueError('Passed model verification receipt lacks its proof: '+identity)
    return True


def controller(runtime, source, selected):
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    import psutil
    api = install_facades(); directory = ROOT/runtime['state_root']; directory.mkdir(parents=True, exist_ok=True)
    lane = api['exclusive_lock'](directory/'controller.lock')
    if lane is None:
        raise RuntimeError('Original complete-cohort statistics queue already live')
    done, failed = set(), {}
    common = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), required_tasks=215,
        required_models=500, required_dataset_cohorts=100, required_paired_comparisons=95,
        required_macro_cohorts=20, original_scope=runtime['original_scope'], all_original_experiments_complete=False)
    def update(**values):
        write(directory/'status.json', dict(common, completed_tasks=sorted(done), completed_task_count=len(done),
            failed_preserved=dict(failed), captured_utc=datetime.now(timezone.utc).isoformat(), **values))
    try:
        while len(done)+len(failed) < len(selected):
            for identity, task in selected.items():
                if identity in done or identity in failed:
                    continue
                try:
                    proof, receipt = stage_paths(runtime, identity)
                    if receipt.exists():
                        read_completed_task(runtime, task); done.add(identity); continue
                    if not readiness(runtime, source, task, done, failed):
                        continue
                    attempt = directory/'attempts'/identity
                    if attempt.exists() or proof.parent.exists():
                        raise ValueError('Incomplete previous statistics attempt preserved; no automatic retry')
                    def waiting(value):
                        update(**dict(value, active_pid=None, task=identity))
                    with api['resource_slot'](ROOT/runtime['resource_lock'], runtime['settings'], waiting, poll_seconds=10):
                        attempt.mkdir(parents=True, exist_ok=False)
                        command = [runtime['python'], '-B', '-X', 'utf8', '-u', '-m', MODULE,
                            '--config', runtime['_config_path'], '--worker', '--task-id', identity]
                        with (attempt/'stdout.log').open('xb') as out, (attempt/'stderr.log').open('xb') as err:
                            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'),
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                            handle = dict(pid=child.pid, create_time=psutil.Process(child.pid).create_time(), command=command)
                            write(attempt/'launch.json', handle)
                            update(state='running_complete_five_seed_'+task['kind'], task=identity, active_pid=child.pid,
                                   active_create_time=handle['create_time'])
                            code, peaks = api['guarded_wait'](child, runtime['settings'], poll_seconds=2)
                        passed = code == 0 and not peaks['resource_guard_aborted'] and proof.exists() and read(proof).get('passed') is True
                        write(receipt, dict(task=identity, passed=passed, exit_code=code, **peaks,
                            proof_sha256=sha(proof) if proof.exists() else None, exact_child_handle=handle))
                        if not passed:
                            raise ValueError('Statistics worker/guard failed; attempt preserved without retry')
                        read_completed_task(runtime, task); done.add(identity)
                except Exception as error:
                    failed[identity] = str(error)
                update(state='scanning_actual_complete_verified_cohorts', task=None, active_pid=None)
            update(state='waiting_all_five_actual_full_model_proofs', task=None, active_pid=None)
            time.sleep(10)
        update(state='all215_statistics_attempts_terminal_other_original_experiments_remain', task=None, active_pid=None)
    finally:
        lane.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--worker', action='store_true'); parser.add_argument('--task-id')
    args = parser.parse_args(); runtime = read(ROOT/args.config)
    source, cases, contract, selected = verify(runtime); runtime['_config_path'] = args.config
    if args.probe_only:
        print(json.dumps(dict(passed=True, required_models=500, tasks=len(selected),
            task_kinds={k:sum(t['kind'] == k for t in selected.values()) for k in ['cohort', 'paired', 'macro']},
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules,
            training_updates_performed=0, complete_cohort_statistics_produced=0, all_original_experiments_complete=False)))
    elif args.worker:
        worker(runtime, source, cases, contract, selected[args.task_id], selected)
    else:
        controller(runtime, source, selected)


if __name__ == '__main__':
    main()
