"""Bind all completed native industrial range metrics to their exact scores.

This verifies existing full TAB calculations and their complete geometry. It
does not independently rerun VUS or affiliation integrals, train, or infer.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import zipfile

from scripts.flow_matching import audit_industrial_full550_independent_metrics_v1 as point
from scripts.flow_matching import audit_industrial_three_recovery_independent_metrics_v1 as recovery
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write

MODULE = 'scripts.flow_matching.audit_industrial_full466_native_ranges_v2'
TEST = 'tests/test_fm_industrial_full466_native_ranges_v2.py'
CONFIG = 'configs/reproducibility/fm_industrial_full466_native_ranges.v2.json'
PRIOR = 'projects/flow_matching_research/results/2026-10-16/industrial_full5_466_canonical_saved_score_metric_audits_v1'
SCIENCES = ['configs/experiments/fm_industrial_tab_range.sparse.v2.json',
            'configs/experiments/fm_recovery_tab_ranges.v2.json']
AFFILIATION = ['affiliation_f', 'affiliation_precision', 'affiliation_recall']
PREVIOUS = 'configs/reproducibility/fm_industrial_full466_native_ranges.v1.json'


def registration():
    paths = [MODULE.replace('.', '/') + '.py', TEST, point.CONFIG, recovery.CONFIG,
             PRIOR + '/packet_integrity.json', PRIOR + '/all466_proof_source_and_guard_bindings.json',
             PRIOR + '/all110_original_model_dataset_seed_group_completion.json', *SCIENCES,
             PREVIOUS, 'scripts/flow_matching/audit_industrial_full466_native_ranges_v1.py',
             'tests/test_fm_industrial_full466_native_ranges_v1.py',
             'experiments/runs/fm_industrial_full466_native_ranges_v1/receipt.json']
    return dict(original_global_scope=read(ROOT/point.CONFIG)['original_global_scope'],
        required_completed_source_cases=466, required_original_CPU_cases=550,
        required_original_GPU_cases=850, required_model_dataset_groups=110,
        complete_five_seed_groups=89, partial_groups=6, empty_groups=15,
        state_root='experiments/runs/fm_industrial_full466_native_ranges_v2',
        artifact_root='F:/aicoding/IIA_Data/experiments/industrial_full466_native_ranges_v2',
        resource_lock='experiments/runs/fm_bounded_transport_math_diagnostic_cpu.lock',
        settings=read(ROOT/point.CONFIG)['settings'],
        independent_threshold_tolerance=read(ROOT/point.CONFIG)['numerical_metric_tolerance'],
        prior_failed_attempt_preserved=read(ROOT/PREVIOUS)['state_root'],
        source_receipts=[dict(path=p, sha256=sha(ROOT/p)) for p in paths],
        full_native_metric_geometry_verified=True, independent_VUS_reexecution=False,
        independent_affiliation_integrals=False, new_model_results=0, new_optimizer_updates=0,
        new_neural_forwards=0, new_training_seed_slots=0, source_historical_resource_gate_certified=False,
        full_TAB_training_harness_equivalence=False, paper_equivalence=False,
        all_original_experiments_complete=False)


def verify(c):
    if c != registration():
        raise ValueError('Frozen full466 sources, original scope and bounded audit guards required')
    integrity = read(ROOT/PRIOR/'packet_integrity.json')['files']
    if not isinstance(integrity, list) or len({r['path'] for r in integrity}) != len(integrity):
        raise ValueError('Original immutable packet inventory required')
    for receipt in integrity:
        if sha(ROOT/PRIOR/receipt['path']) != receipt['sha256']:
            raise ValueError('Original independent point-metric packet changed')
    pc = read(ROOT/point.CONFIG)
    q, manifest = point.verify(pc)
    rc = read(ROOT/recovery.CONFIG)
    _, _, _, recovered = recovery.verify(rc)
    jobs = {j['id']: j for j in q['jobs']}
    executions = dict(jobs)
    for item in recovered:
        executions[item['original_id']] = recovery.validate_identity(item, jobs)
    bindings = read(ROOT/PRIOR/'all466_proof_source_and_guard_bindings.json')['records']
    if len(bindings) != 466 or sum(b['separate_recovery'] for b in bindings.values()) != 3:
        raise ValueError('All463 direct and three mapped source cases required')
    return q, manifest, jobs, executions, bindings


def geometry(labels):
    if not labels or any(x not in (0, 1) for x in labels):
        raise ValueError('Complete nonempty binary entity labels required')
    lengths, start = [], None
    for index, value in enumerate([*labels, 0]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            lengths.append(index-start)
            start = None
    return dict(test_points=len(labels), positives=sum(labels), events=len(lengths),
        max_buffer=2*int(statistics.median(lengths)) if lengths and sum(labels)<len(labels) else None)


def defined_macro(records, metrics):
    result = {}
    for field in metrics:
        values = [r[field] for r in records.values() if r[field] is not None]
        if any(not math.isfinite(x) or not 0 <= x <= 1.000000000001 for x in values):
            raise ValueError('Finite bounded native metrics or explicit undefined values required')
        result[field] = dict(defined_entities=len(values), undefined_entities=len(records)-len(values),
            mean_over_defined_entities=statistics.mean(values) if values else None)
    return result


def equal_macro(records, actual, metrics):
    expected = defined_macro(records, metrics)
    if set(actual) != set(expected):
        raise ValueError('All metric macro fields required')
    for key, value in expected.items():
        item = actual[key]
        if (set(item) != set(value) or item['defined_entities'] != value['defined_entities']
                or item['undefined_entities'] != value['undefined_entities']):
            raise ValueError('Undefined entities must remain explicit in the macro denominator')
        left, right = item['mean_over_defined_entities'], value['mean_over_defined_entities']
        if (left is None) != (right is None) or (left is not None and not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError('Full independent entity macro differs')


def validate_ranges(e, source, science, shapes):
    result_digest = hashlib.sha256(json.dumps(science, sort_keys=True).encode()).hexdigest()
    identity = hashlib.sha256((e['model_result_sha256']+result_digest).encode()).hexdigest()
    if (e['status'] != 'completed_range_evaluation' or e['id'] != source['id']
            or e['dataset'] != source['dataset'] or e['model_config'] != source['model_config']
            or e['seed'] != source['seed'] or e['scores_sha256'] != source['scores_sha256']
            or e['evaluation_config_sha256'] != result_digest or e['evaluation_identity'] != identity
            or e['source_receipts'] != science['source_receipts'] or e['tab_commit'] != science['tab_commit']):
        raise ValueError('Exact frozen source result, native evaluation rules and scoring identity required')
    entity_fields = ['VUS_per_entity', 'strict_affiliation_per_entity', 'tab_ratio_affiliation_per_entity']
    if any(set(e[field]) != set(shapes) for field in entity_fields):
        raise ValueError('Every original entity must be present')
    strict = source['metrics']['strict']
    if set(strict) != {'0.1', '1', '5'} or set(e['strict_affiliation_entity_macro']) != set(strict):
        raise ValueError('All three original frozen validation thresholds required')
    for ratio, item in strict.items():
        cal = item['calibration']
        if (cal['calibration_split'] != 'validation' or cal['false_alarm_percent'] != float(ratio)
                or cal['rule'] != 'validation_score_percentile_linear_strict_greater'):
            raise ValueError('Primary validation thresholds cannot be replaced by test calibration')
    grid = source['metrics']['tab_core_diagnostic']['ratio_grid']
    if len(grid) != 10 or len(e['tab_ratio_affiliation_entity_macro']) != len(grid):
        raise ValueError('Complete original ten-ratio diagnostic grid required')
    for entity, shape in shapes.items():
        item = e['VUS_per_entity'][entity]
        if shape['max_buffer'] is None:
            if (item['status'] != 'undefined_single_class_reference' or item['VUS_ROC'] is not None
                    or item['VUS_PR'] is not None or item['max_buffer'] is not None):
                raise ValueError('Single-class VUS must remain undefined')
        elif (item['status'] != 'defined' or item['max_buffer'] != shape['max_buffer']
                or item['buffers_evaluated'] != shape['max_buffer']+1 or item['threshold_samples'] != 250):
            raise ValueError('Every integer buffer and all250 native thresholds required')
        if set(e['strict_affiliation_per_entity'][entity]) != set(strict):
            raise ValueError('All original entity validation controls required')
        if len(e['tab_ratio_affiliation_per_entity'][entity]) != len(grid):
            raise ValueError('No original entity diagnostic ratio may be omitted')
        affiliation = [*e['strict_affiliation_per_entity'][entity].values(), *e['tab_ratio_affiliation_per_entity'][entity]]
        for record in affiliation:
            values = [record[k] for k in AFFILIATION]
            all_defined = all(x is not None for x in values)
            if ((record['status'] == 'defined') != all_defined
                    or record['status'] not in {'defined', 'undefined_reference_nan', 'undefined_no_ground_truth_event'}
                    or (shape['events'] == 0 and (record['status'] != 'undefined_no_ground_truth_event' or any(x is not None for x in values)))):
                raise ValueError('Affiliation undefined-event and undefined-prediction semantics required')
    equal_macro(e['VUS_per_entity'], e['VUS_entity_macro'], ['VUS_ROC', 'VUS_PR'])
    for ratio in strict:
        equal_macro({k:v[ratio] for k,v in e['strict_affiliation_per_entity'].items()},
            e['strict_affiliation_entity_macro'][ratio], AFFILIATION)
    for index, item in enumerate(grid):
        rows = [e['tab_ratio_affiliation_entity_macro'][index],
                *[v[index] for v in e['tab_ratio_affiliation_per_entity'].values()]]
        if any(r['ratio_percent'] != item['ratio_percent'] or r['threshold'] != item['threshold'] for r in rows):
            raise ValueError('All native diagnostic thresholds must match the unchanged source')
        equal_macro({k:v[index] for k,v in e['tab_ratio_affiliation_per_entity'].items()},
            rows[0]['metrics'], AFFILIATION)


def complete_stats(values):
    if len(values) != 5:
        raise ValueError('Only complete five-model seed groups produce summaries')
    defined = [x for x in values if x is not None]
    if any(not math.isfinite(x) for x in defined):
        raise ValueError('Nonfinite values cannot enter seed summaries')
    if len(defined) != 5:
        return dict(status='complete_group_metric_undefined_for_some_seeds', n=5,
            defined_seed_count=len(defined), mean=None, sample_std=None, t95_low=None, t95_high=None)
    sd = statistics.stdev(defined)
    mean = statistics.mean(defined)
    half = 2.7764451051977987 * sd / math.sqrt(5)
    return dict(status='complete_five_seed_summary', n=5, defined_seed_count=5,
        mean=mean, sample_std=sd, t95_low=mean-half, t95_high=mean+half)


def verify_independent_threshold(independent, original, tolerance):
    # Native TAB input remains the exact source threshold. Only comparison
    # with the independently reconstructed percentile uses its frozen bounds.
    if (not math.isfinite(independent) or not math.isfinite(original)
            or not math.isclose(independent,original,rel_tol=tolerance['rtol'],abs_tol=tolerance['atol'])):
        raise ValueError('Previously independently checked calibration changed')


def worker(c):
    q, manifest, jobs, executions, bindings = verify(c)
    cache = {}
    def check(path, digest):
        key = str(Path(path).resolve())
        if key not in cache:
            cache[key] = sha(path)
        if cache[key] != digest:
            raise ValueError('Frozen source or full native artifact changed: '+str(path))
    shapes = {}
    for dataset in manifest['datasets']:
        path = ROOT/dataset['input_npz']; check(path, dataset['input_sha256'])
        for raw in dataset['source_files']:
            check(ROOT/raw['path'], raw['sha256'])
        entities = {}
        with zipfile.ZipFile(path) as archive:
            for entity in dataset['entities']:
                header, payload = point.archive_array(archive, entity['array_prefix']+'_labels')
                labels = point.vector(header, payload, entity['test_points'], '|u1')
                entities[entity['entity_id']] = dict(geometry(labels), label_payload_sha256=hashlib.sha256(payload).hexdigest(),
                    label_array=entity['array_prefix']+'_labels')
        shapes[dataset['id']] = entities
    sciences = [read(ROOT/p) for p in SCIENCES]
    for science in sciences:
        for receipt in science['source_receipts']:
            check(ROOT/receipt['path'], receipt['sha256'])
    records, rows, entity_rows = {}, [], []
    for cid, binding in bindings.items():
        job = executions[cid]; source_path = ROOT/binding['source_result_path']
        check(source_path, binding['source_result_sha256']); source = read(source_path)
        if source['job'] != job or source['id'] != binding['actual_execution_id']:
            raise ValueError('Canonical recovery mapping differs')
        for receipt in job['frozen_files']:
            check(ROOT/receipt['path'], receipt['sha256'])
        check(source_path.parent/'model.pt', binding['checkpoint_sha256'])
        check(source_path.parent/'scores.npz', binding['scores_sha256'])
        check(binding['guard_path'], binding['guard_sha256']); guard = read(binding['guard_path'])
        if not guard['passed'] or guard['exit_code'] != 0 or guard['resource_guard_aborted']:
            raise ValueError('Original independent score audit must have passed')
        if binding['separate_recovery']:
            check(binding['independent_report_container'], binding['independent_report_container_sha256'])
            proof = read(binding['independent_report_container'])['records'][cid]['independent_report']
        else:
            check(binding['independent_report_path'], binding['independent_report_sha256'])
            proof = read(binding['independent_report_path'])
        if (not proof['passed'] or proof['source_result_sha256'] != sha(source_path)
                or proof['checkpoint_sha256'] != binding['checkpoint_sha256']
                or proof['scores_sha256'] != binding['scores_sha256']):
            raise ValueError('Complete independent score provenance required')
        science = sciences[int(binding['separate_recovery'])]
        path = ROOT/science['output_root']/'jobs'/job['id']/'evaluation.json'
        digest = path.with_name('evaluation.sha256').read_text(encoding='ascii').strip()
        check(path, digest); e = read(path)
        if e['model_result_sha256'] != binding['source_result_sha256'] or e['model_result_path'] != binding['source_result_path']:
            raise ValueError('Native metric source file identity differs')
        validate_ranges(e, source, science, shapes[job['dataset']])
        old = e.get('reused_original_evaluation')
        if old:
            check(ROOT/old['path'], old['sha256']); original = read(ROOT/old['path'])
            changed = {'evaluation_identity','evaluation_config_sha256','source_receipts','reused_original_evaluation','execution_engine'}
            if (not old['all_metric_values_and_original_elapsed_time_preserved']
                    or {k:v for k,v in e.items() if k not in changed} != {k:v for k,v in original.items() if k not in changed}):
                raise ValueError('Every originally computed native value and elapsed time must be preserved')
        records[cid] = dict(actual_execution_id=job['id'], native_evaluation_path=path.relative_to(ROOT).as_posix(),
            native_evaluation_sha256=digest, source_result_sha256=binding['source_result_sha256'],
            scores_sha256=binding['scores_sha256'], independent_point_proof_binding=binding,
            reused_original_native_calculation=bool(old), full_entities=len(shapes[job['dataset']]))
        for ratio in ['0.1','1','5']:
            sm = proof['independent_strict_metrics'][ratio]
            verify_independent_threshold(sm['threshold'],source['metrics']['strict'][ratio]['calibration']['threshold'],
                c['independent_threshold_tolerance'])
            values = {**e['VUS_entity_macro'], **e['strict_affiliation_entity_macro'][ratio]}
            for metric, v in values.items():
                rows.append(dict(canonical_id=cid, actual_execution_id=job['id'], model_config=job['model_config'],
                    dataset=job['dataset'], seed=job['seed'], validation_alarm_percent=ratio,
                    actual_primary_evaluation=ratio=='1', metric=metric, value=v['mean_over_defined_entities'],
                    defined_entities=v['defined_entities'], undefined_entities=v['undefined_entities'],
                    independent_point_F1=sm['micro']['f1'], independent_point_AUROC=sm['micro']['auroc']))
            for entity, shape in shapes[job['dataset']].items():
                bound = proof['all_array_bindings'][shape['label_array']]
                if bound['sha256'] != shape['label_payload_sha256'] or bound['shape'] != [shape['test_points']]:
                    raise ValueError('Every native label coordinate must match the independent score proof')
                entity_rows.append(dict(canonical_id=cid, model_config=job['model_config'], dataset=job['dataset'], seed=job['seed'],
                    validation_alarm_percent=ratio, entity=entity, test_points=shape['test_points'], positive_points=shape['positives'],
                    events=shape['events'], max_buffer=shape['max_buffer'],
                    max_buffer_to_test_length=shape['max_buffer']/shape['test_points'] if shape['max_buffer'] is not None else None,
                    VUS_ROC=e['VUS_per_entity'][entity]['VUS_ROC'], VUS_PR=e['VUS_per_entity'][entity]['VUS_PR'],
                    **{k:e['strict_affiliation_per_entity'][entity][ratio][k] for k in AFFILIATION}))
    groups = defaultdict(list)
    for job in q['jobs']:
        groups[job['model_config'],job['dataset']].append(job)
    index = {(r['canonical_id'],r['validation_alarm_percent'],r['metric']):r for r in rows}
    summaries, group_status = [], []
    for (model,dataset), group in sorted(groups.items()):
        if sorted(j['seed'] for j in group) != [1103,1104,1105,1106,1107]:
            raise ValueError('Original complete five-seed matrix required')
        done = [j for j in group if j['id'] in records]
        group_status.append(dict(model_config=model,dataset=dataset,complete_seeds=[j['seed'] for j in done],
            required_seeds=[1103,1104,1105,1106,1107],complete=len(done)==5,partial_statistics_computed=False))
        if len(done) != 5:
            continue
        for ratio in ['0.1','1','5']:
            for metric in ['VUS_ROC','VUS_PR',*AFFILIATION]:
                rs = [index[j['id'],ratio,metric] for j in done]
                summaries.append(dict(model_config=model,dataset=dataset,validation_alarm_percent=ratio,
                    actual_primary_evaluation=ratio=='1',metric=metric,**complete_stats([r['value'] for r in rs])))
    if (len(records)!=466 or len(rows)!=6990 or len(group_status)!=110
            or sum(x['complete'] for x in group_status)!=89 or len(summaries)!=1335
            or sum(0<len(x['complete_seeds'])<5 for x in group_status)!=6
            or sum(not x['complete_seeds'] for x in group_status)!=15):
        raise ValueError('All466 completed cases and all89 complete five-seed groups required')
    out = Path(c['artifact_root']);out.mkdir(parents=True,exist_ok=False)
    for name, data in [('all466_native_range_controls.csv',rows),('all466_full_entity_geometry_and_native_metrics.csv',entity_rows),
                       ('all89_complete_five_seed_native_range_summaries.csv',summaries)]:
        with (out/name).open('x',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    write(out/'all466_native_range_proof_bindings.json',dict(records=records))
    write(out/'all110_original_group_completion.json',dict(groups=group_status))
    write(out/'all5_full_native_entity_geometry.json',dict(datasets=shapes))
    report = dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),cases=466,native_metric_controls=6990,
        full_entity_controls=len(entity_rows),complete_groups=89,full_five_seed_metric_summaries=1335,
        partial_groups=6,empty_groups=15,remaining_original_CPU_cases=84,
        originally_computed_native_values_preserved=True,full250thresholds_and_all_integer_buffers=True,
        every_original_entity_and_label_coordinate_verified=True,undefined_values_remain_null=True,
        all_three_validation_controls_verified=True,actual_primary='validation1percent_noPA',
        diagnostic_test_pool_ten_ratio_controls_verified=True,test_pool_controls_are_not_primary=True,
        independent_VUS_reexecution=False,independent_affiliation_integrals=False,
        source_historical_resource_gate_certified=False,new_complete_model_results=0,new_training_seed_slots=0,
        new_optimizer_updates=0,new_neural_forwards=0,full_TAB_training_harness_equivalence=False,
        original_global_scope=c['original_global_scope'],all_original_experiments_complete=False)
    write(out/'report.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register',action='store_true');parser.add_argument('--probe-only',action='store_true')
    parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.register:
        with (ROOT/CONFIG).open('x',encoding='utf-8') as f:json.dump(registration(),f,indent=2)
        return
    c=read(ROOT/CONFIG);verify(c)
    if args.probe_only:
        print(json.dumps(dict(passed=True,numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,
            complete_source_cases=466,full_original_CPU_scope=550,new_model_results=0)));return
    if args.worker:worker(c);return
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=install_facades();base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Full466 native audit already owned')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None)
    def update(v):
        v=dict(v)
        if v.get('state')=='replaying_full_checkpoints':v['state']='verifying_all466_full_native_ranges'
        state.update(v,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        if (base/'worker.stdout.log').exists():raise RuntimeError('Preserve existing terminal attempt')
        with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],update,poll_seconds=10):
            command=[sys.executable,'-B','-X','utf8','-u','-m',MODULE,'--worker']
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                update(dict(state='verifying_all466_full_native_ranges',active_pid=child.pid,
                    active_create_time=psutil.Process(child.pid).create_time(),actual_command=command))
                code,usage=capped_wait(child,c,api,update)
            path=Path(c['artifact_root'])/'report.json'
            passed=code==0 and not usage['resource_guard_aborted'] and path.exists() and read(path)['passed']
            write(base/'receipt.json',dict(passed=passed,exit_code=code,**usage,proof_sha256=sha(path) if path.exists() else None))
            if not passed:raise RuntimeError('Terminal full466 native range audit failure preserved')
        update(dict(state='complete',active_pid=None,passed=True))
    except Exception as error:
        update(dict(state='terminal_failure_preserved',active_pid=None,error=str(error)));raise
    finally:owner.close()


if __name__=='__main__':main()
