"""Register all100five-seed,95paired and20macro reconstruction summaries."""
import json

from scripts.flow_matching import run_anomaly_transformer_full_cohort_statistics_v1 as runner

ROOT, read, sha = runner.ROOT, runner.read, runner.sha
CONFIG = 'configs/runtime/fm_anomaly_transformer_full_cohort_statistics.v1.json'


def registration():
    source = read(ROOT/runner.PRODUCER)
    cases = {}
    for family, (path, _, _) in runner.producer.FAMILIES.items():
        config = read(ROOT/path)
        for job in config['jobs']:
            cases[job['id']] = dict(family=family, path=path, config=config, job=job)
    selected = runner.tasks(cases)
    files = [p['path'] for p in source['source_receipts']]+[
        runner.PRODUCER,
        'scripts/flow_matching/run_anomaly_transformer_full_cohort_statistics_v1.py',
        'scripts/flow_matching/register_anomaly_transformer_full_cohort_statistics_v1.py',
        'scripts/flow_matching/summarize_saits_full_five_seed_cohort_v1.py',
        'tests/test_fm_anomaly_transformer_full_cohort_statistics_v1.py']
    return dict(schema_version=1, producer=runner.PRODUCER, producer_sha256=sha(ROOT/runner.PRODUCER),
        python=source['python'], settings=source['settings'], resource_lock=source['resource_lock'],
        original_scope=source['original_scope'], required_models=500, required_tasks=215,
        seeds=runner.SEEDS, datasets=runner.DATASETS, confidence_level=.95,
        student_t_critical_df4_975=runner.DF4_CRITICAL_975,
        student_t_critical_reference='scripts/flow_matching/summarize_saits_full_five_seed_cohort_v1.py',
        paired_test_support='common_frozen_covered_prefix',
        paired_calibration='each_model_original_full_validation_scores',
        macro_weighting='equal_dataset_weight_within_each_seed',
        primary_endpoint='validation_ratio.no_PA.f1', fixed_delta_controls_are_diagnostic=True,
        interval_scope='descriptive_training_seed_variability_only_unclipped',
        tasks=list(selected.values()),
        artifact_root='F:/aicoding/IIA_Data/experiments/anomaly_transformer_full_cohort_statistics_v1',
        state_root='experiments/runs/fm_anomaly_transformer_full_cohort_statistics_v1',
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(set(files))],
        boundary='Only complete five-original-seed cohorts after full checkpoint and independent-array '
            'proofs with passed original/replay guards.100dataset cohorts cover every original score '
            'stream/control with mean,sampleSD,unclippeddf4tCI,median,range and leave-one-seed-out means. '
            '95variant-vs-main comparisons independently recompute primary scores on common frozen '
            'covered test prefixes,retaining each original complete validation calibration and every seed. '
            '20macro summaries average all five datasets within each seed before seed uncertainty. '
            'No pooledPA/noPA scores,no test-selected criterion,no missing-seed deletion,no multiplicity '
            'or dataset-generalization guarantee,no synthetic benchmark,no model inference/training/retry. '
            'All32GiBphysical/40GiBcommit/16GiBemergency safeguards and full45/681/273minimumscope retained.')


def main():
    target = ROOT/CONFIG
    if target.exists():
        raise FileExistsError('Preserve the frozen full cohort statistics registration')
    runtime = registration(); runner.verify(runtime)
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(runtime, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(configuration=CONFIG, required_unique_models=500,
        dataset_cohorts=100, paired_comparisons=95, five_dataset_macro_cohorts=20,
        statistics_tasks=215, numerical_workers_started=0)))


if __name__ == '__main__':
    main()
