"""Freeze all206 source-bound five-seed local objective analyses."""
import json
from scripts.flow_matching import run_objective_full440_statistics_v1 as runner

CONFIG = 'configs/runtime/fm_objective_full440_statistics.v1.json'


def registration():
    source = runner.read(runner.ROOT/runner.PRODUCER)
    cases = runner.producer.verify(source)
    pins = [p['path'] for p in source['source_receipts']] + [runner.PRODUCER,
        'scripts/flow_matching/run_objective_full440_statistics_v1.py',
        'scripts/flow_matching/register_objective_full440_statistics_v1.py',
        'scripts/flow_matching/resume_anomaly_transformer_full500_metadata_v1.py',
        'tests/test_fm_objective_full440_statistics_v1.py']
    return dict(schema_version=1,producer=runner.PRODUCER,producer_sha256=runner.sha(runner.ROOT/runner.PRODUCER),
        required_cases=440,required_tasks=206,seeds=runner.SEEDS,datasets=runner.DATASETS,
        tasks=list(runner.tasks(cases).values()),settings=source['settings'],resource_lock=source['resource_lock'],
        original_scope=source['original_scope'],student_t_df4_975=runner.T_CRITICAL,
        primary_endpoint='strict.1.micro.f1',
        python=str(runner.ROOT/runner.read(runner.ROOT/runner.producer.QUEUES['original'][0])['python']),
        artifact_root='F:/aicoding/IIA_Data/experiments/objective_full440_statistics_v1',
        state_root='experiments/runs/fm_objective_full440_statistics_v1',
        source_receipts=[dict(path=p,sha256=runner.sha(runner.ROOT/p)) for p in sorted(set(pins))],
        boundary='All88five-seed cohorts cover440distinct original/repair/preprocessing cases. '
            '107paired analyses=80variant-versus-independent+11preprocessing+16SF2Mscore-head controls. '
            '11macros average all7datasets inside each matching seed before five-seed uncertainty. '
            'Every positive case requires unchanged full native/independent artifacts and passed fresh guards. '
            'Original105failures stay negative outcomes; repairs never replace or pool them. '
            'Actual terminal source/replay failures make affected analyses unestimable,not successful. '
            'Mean/sampleSD/SE/unclippeddf4t95interval/median/range/leave-one-seed-out are descriptive '
            'training-seed variability only. Undefined endpoints retain all5slots and receive no partial CI. '
            'Primary strict1%validation-only noPAF1 remains separate from three validation ratios,events, '
            'ranking,ten test-pooled noPA/PA controls and test-label-selected diagnostic maxima. '
            'Pairing requires same seed,dataset,full split coverage,labels and available row/timestamp/fault metadata; '
            'each model keeps its original full validation calibration. No training/inference/retry, '
            'preprocessing-value reconstruction,VUS/affiliation integral,causal deployment,fullTAB, '
            'historicalpaper equivalence or full45/681/273completion certified. '
            'Original16/32/16GiBguards and10/12GiBcaps retained;status-only boundedWindows rename policy reused.')


def main():
    target = runner.ROOT/CONFIG
    if target.exists(): raise FileExistsError('Preserve frozen full statistics registration')
    config = registration(); runner.verify(config)
    with target.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(config,stream,indent=2); stream.write('\n')
    print(json.dumps(dict(configuration=CONFIG,required_cases=440,required_tasks=206,numerical_workers_started=0)))


if __name__ == '__main__': main()
