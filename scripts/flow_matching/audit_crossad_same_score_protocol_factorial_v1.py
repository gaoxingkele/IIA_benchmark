"""All139 fixed-score calibration x point-adjustment protocol diagnostics.

Use identical complete scores for every cell of a prespecified2x2 design at
each of the three frozen alarm budgets. This is not a full TAB execution.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics
from types import FunctionType
import zipfile

from scripts.flow_matching import audit_crossad_full139_independent_metrics_v1 as parent
from scripts.flow_matching import collect_crossad_CUDA_full139_verified_v1 as groups
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write

MODULE = 'scripts.flow_matching.audit_crossad_same_score_protocol_factorial_v1'
CONFIG = 'configs/reproducibility/fm_crossad_same_score_protocol_factorial.v1.json'
TEST = 'tests/test_fm_crossad_same_score_protocol_factorial_v1.py'
PROTOCOLS = ['validation_raw', 'validation_PA', 'pooled_train_test_raw', 'pooled_train_test_PA']
DELTAS = ['PA_at_validation', 'pooled_calibration_without_PA', 'PA_at_pooled_calibration', 'combined', 'interaction']


def registration():
    c = copy.deepcopy(parent.registration())
    c.update(upstream=parent.CONFIG, state_root='experiments/runs/fm_crossad_same_score_protocol_factorial_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/crossad_same_score_protocol_factorial_v1',
        alarm_budget_percent=[.1, 1., 5.], primary_alarm_budget_percent=1.,
        protocols=PROTOCOLS, primary_training_seeds=groups.SEEDS, required_primary_cohorts=22,
        calibration_arithmetic='Same independent float64 linear quantile and strict greater-than in all four cells; not source TAB float32 identity certification',
        metrics=['precision', 'recall', 'f1', 'exact_confusion_counts', 'paired_F1_protocol_differences'],
        new_model_training_updates=0, new_neural_inference_executed=False,
        test_label_threshold_selection=False, point_adjustment_uses_test_event_labels=True,
        original_source_concatenated_timeline=True, strict_TAB_result=False, paper_equivalence=False)
    pins = {x['path']:x['sha256'] for x in c['source_receipts']}
    for p in [parent.CONFIG, MODULE.replace('.', '/')+'.py', TEST,
              'scripts/flow_matching/collect_crossad_CUDA_full139_verified_v1.py']:
        pins[p] = sha(ROOT/p)
    c['source_receipts'] = [dict(path=p, sha256=h) for p,h in sorted(pins.items())]
    return c


def verify(c):
    if c != registration():
        raise ValueError('Frozen full139 same-score factorial protocol changed')
    upstream = read(ROOT/parent.CONFIG)
    q, _ = parent.verify(upstream)
    if len(groups.cohorts(q['jobs'], {})['primary_cohorts']) != 22:
        raise ValueError('All22 original primary five-seed groups required')
    return q, upstream


def point_adjust(labels, prediction):
    if len(labels) != len(prediction) or any(v not in (0, 1) for v in labels):
        raise ValueError('Every full aligned binary test label/prediction required')
    result = list(prediction)
    first = 0
    while first < len(labels):
        if not labels[first]:
            first += 1
            continue
        last = first + 1
        while last < len(labels) and labels[last]:
            last += 1
        if any(prediction[first:last]):
            result[first:last] = [True] * (last-first)
        first = last
    return result


def confusion(labels, prediction):
    tp = sum(bool(y and p) for y,p in zip(labels,prediction))
    fp = sum(bool(not y and p) for y,p in zip(labels,prediction))
    fn = sum(labels) - tp
    return dict(true_positives=tp, false_positives=fp, false_negatives=fn,
        precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn) if tp+fn else 0.,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def factorial(training, validation, test, labels, budgets):
    if not training or not validation or len(test) != len(labels) or not test:
        raise ValueError('Every original full train/validation/test split required')
    sorted_val = sorted(validation)
    sorted_pool = sorted(list(training)+list(test))
    ratios = {}
    for ratio in budgets:
        rows = {}
        for split, values in [('validation', sorted_val), ('pooled_train_test', sorted_pool)]:
            threshold = parent.percentile(values, ratio)
            pred = [v > threshold for v in test]
            for adjusted, final in [(False, pred), (True, point_adjust(labels, pred))]:
                name = split + ('_PA' if adjusted else '_raw')
                rows[name] = dict(threshold=threshold, **confusion(labels, final),
                    test_scores_used_in_calibration=split!='validation',
                    test_labels_used_to_select_threshold=False, test_labels_used_in_postprocessing=adjusted)
        a,b,c,d = [rows[k]['f1'] for k in PROTOCOLS]
        ratios[format(ratio, 'g')] = dict(protocols=rows, paired_F1_differences=dict(
            PA_at_validation=b-a, pooled_calibration_without_PA=c-a,
            PA_at_pooled_calibration=d-c, combined=d-a, interaction=(d-c)-(b-a)))
    return ratios


def audit(c, q, upstream, job):
    canonical = job['canonical_CPU_condition_reference']
    source_proof = Path(upstream['artifact_root'])/'conditions'/(canonical+'.json')
    prior = read(source_proof)
    guard_path = ROOT/upstream['state_root']/'guards'/(canonical+'.json')
    guard = read(guard_path)
    if (not prior['passed'] or not guard['passed'] or guard['exit_code'] != 0
            or guard['resource_guard_aborted'] or guard['report_sha256'] != sha(source_proof)):
        raise ValueError('Complete native replay and independent full metric audit required')
    source_job = read(ROOT/q['delegated_first_configuration'])['job'] if canonical==q['delegated_first_original_condition'] else job
    result_path = ROOT/source_job['output_directory']/'result.json'
    archive_path = result_path.parent/'native_scores.npz'
    if sha(result_path) != prior['source_result_sha256'] or sha(archive_path) != prior['full_score_archive_sha256']:
        raise ValueError('Identical original complete model scores required for every protocol')
    counts = prior['complete_points']
    payload, hashes = {}, {}
    with zipfile.ZipFile(archive_path) as archive:
        if set(archive.namelist()) != {'train.npy','validation.npy','test.npy','labels.npy','source_rows.npy'}:
            raise ValueError('Every complete original score/label/timepoint array required')
        for name in ['train','validation','test','labels','source_rows']:
            size = counts.get(name, counts['test'])
            dtype = '|u1' if name=='labels' else '<i8' if name=='source_rows' else '<f4'
            with archive.open(name+'.npy') as stream:
                payload[name], hashes[name] = parent.vector(stream, size, dtype)
    if hashes != prior['all_score_label_and_timestamp_payload_sha256']:
        raise ValueError('Every full score, label and original row byte must match the prior audit')
    if any(row != source_job['train_length']+i for i,row in enumerate(payload['source_rows'])):
        raise ValueError('Original full row order and all tail positions required')
    rows = factorial(payload['train'],payload['validation'],payload['test'],payload['labels'],c['alarm_budget_percent'])
    for ratio, row in rows.items():
        expected = prior['independent_strict_metrics'][ratio]
        parent.equal_fields(dict(threshold=row['protocols']['validation_raw']['threshold']),
            dict(threshold=expected['threshold']), c['numerical_metric_tolerance'])
        parent.equal_fields(row['protocols']['validation_raw'],dict(expected['micro'],
            threshold=expected['threshold'],test_scores_used_in_calibration=False,
            test_labels_used_to_select_threshold=False,test_labels_used_in_postprocessing=False),c['numerical_metric_tolerance'])
    return dict(passed=True, canonical_CPU_condition_reference=canonical,
        dataset=job['dataset'], seed=job['seed'], mode=job['mode'], ablation_row=job.get('ablation_row'),
        raw_sha256=job['raw_sha256'], complete_points=counts,
        source_independent_metric_proof_sha256=sha(source_proof), complete_score_archive_sha256=sha(archive_path),
        all_score_label_and_timestamp_payload_sha256=hashes, protocol_factorial=rows,
        original_source_concatenated_timeline=True, full_original_tail_coverage=True,
        calibration_arithmetic=c['calibration_arithmetic'], new_model_training_updates=0,
        new_neural_inference_executed=False, strict_TAB_result=False, paper_equivalence=False,
        independent_bootstrap_recomputed=False, all_original_experiments_complete=False)


def cohort_summary(q, verified):
    planned = {}
    for job in q['jobs']:
        if job['mode']=='fresh_train' and job['seed'] in groups.SEEDS:
            planned.setdefault(groups.cohort_key(job),{})[job['seed']] = job['canonical_CPU_condition_reference']
    result = {}
    for key, seeds in sorted(planned.items()):
        if set(seeds) != set(groups.SEEDS):
            raise ValueError('Exactly all five original seeds must remain planned')
        present = {seed:verified[identity] for seed,identity in seeds.items() if identity in verified}
        row = dict(verified_training_seeds=sorted(present),complete=len(present)==5)
        if row['complete']:
            signature = {json.dumps(dict(raw=x['raw_sha256'],counts=x['complete_points'],
                labels=x['all_score_label_and_timestamp_payload_sha256']['labels'],
                rows=x['all_score_label_and_timestamp_payload_sha256']['source_rows']),sort_keys=True) for x in present.values()}
            if len(signature) != 1:
                raise ValueError('All five paired model seeds must share exactly the same full dataset/labels/rows')
            row['by_alarm_budget'] = {}
            for ratio in ['0.1','1','5']:
                row['by_alarm_budget'][ratio] = dict(
                    protocol_F1={p:groups.interval([present[s]['protocol_factorial'][ratio]['protocols'][p]['f1'] for s in groups.SEEDS]) for p in PROTOCOLS},
                    paired_F1_differences={d:groups.interval([present[s]['protocol_factorial'][ratio]['paired_F1_differences'][d] for s in groups.SEEDS]) for d in DELTAS})
        result[key] = row
    if len(result) != 22:
        raise ValueError('All22 primary original main/ablation groups required')
    return dict(cohorts=result,complete_primary_cohorts=sum(x['complete'] for x in result.values()),
        partial_cohorts_never_produce_means_or_intervals=True,
        interval_scope='Descriptive paired differences across five fitted model seeds on one frozen source file;not unseen entities or algorithm superiority',
        full_TAB_certification=False, all_original_experiments_complete=False)


def main():
    def write_with_cohorts(path, value):
        write(path,value)
        cpath = ROOT/CONFIG
        if cpath.exists():
            c = read(cpath)
            if Path(path) == ROOT/c['state_root']/'status.json' and 'verified_ids' in value:
                q = read(ROOT/c['queue'])
                verified = {i:read(Path(c['artifact_root'])/'conditions'/(i+'.json')) for i in value['verified_ids']}
                write(Path(c['artifact_root'])/'all22_factorial_cohorts.json',cohort_summary(q,verified))
    function = FunctionType(parent.main.__code__,dict(parent.main.__globals__,
        MODULE=MODULE,CONFIG=CONFIG,registration=registration,verify=verify,audit=audit,write=write_with_cohorts))
    function()


if __name__=='__main__':
    main()
