"""Independently verify full saved precision diagnostics and original SMD seed1105."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from scripts.flow_matching.prepare_tsad_execution import sha
from scripts.flow_matching.run_tsad_queue import complete_result
from iia_benchmark.evaluation.mtsad_strict import calibrate_validation, evaluate_entities


def check_arrays(inputs, saved, dataset):
    expected, lengths, validation, entities = set(), {}, [], {}
    for entry in dataset['entities']:
        prefix, entity = entry['array_prefix'], entry['entity_id']
        for split in ('train', 'validation', 'test'):
            key = prefix + '_' + split
            expected.add(key)
            values = saved[key]
            if values.shape != (len(inputs[key]),) or not np.isfinite(values).all():
                raise ValueError('Full finite timestamp scores required: ' + key)
        key = prefix + '_labels'
        expected.add(key)
        if not np.array_equal(saved[key], inputs[key]) or len(saved[key]) != entry['test_points']:
            raise ValueError('Original full labels differ: ' + key)
        lengths[entity] = len(saved[key])
        entities[entity] = (saved[key], saved[prefix + '_test'])
        validation.append(saved[prefix + '_validation'])
    if set(saved.files) != expected:
        raise ValueError('Full array identity differs')
    return entities, lengths, np.concatenate(validation)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    target = ROOT / 'projects/flow_matching_research/results/2026-10-11/tranad_reference_float64_v1'
    config_path = ROOT / 'configs/reproducibility/fm_tranad_float64_existing_checkpoint_diagnostic.2026-10-11_v1.json'
    config = json.loads(config_path.read_text('utf-8'))
    for source in config['source_receipts']:
        if sha(ROOT / source['path']) != source['sha256']:
            raise ValueError('Diagnostic frozen source differs')
    report_path = ROOT / config['output_directory'] / 'precision_diagnostic.json'
    report = json.loads(report_path.read_text('utf-8'))
    state_path = ROOT / config['controller_directory'] / 'status.json'
    state = json.loads(state_path.read_text('utf-8'))
    if state['state'] != 'completed' or state['exit_code'] != 0 or state['resource_guard_aborted']:
        raise ValueError('Actual full diagnostic did not finish successfully')
    if state['config_sha256'] != sha(config_path):
        raise ValueError('Terminal precision configuration differs')
    manifest = json.loads((ROOT / config['dataset_manifest']).read_text('utf-8'))
    dataset = next(d for d in manifest['datasets'] if d['id'] == 'smap')
    if sha(ROOT / dataset['input_npz']) != dataset['input_sha256']:
        raise ValueError('Original full SMAP input changed')
    cases = {c['job']['id']: c for c in config['cases']}
    if len(report['records']) != 5 or {r['original_float32_job_id'] for r in report['records']} != set(cases):
        raise ValueError('All five original checkpoints required')
    proofs = []
    with np.load(ROOT / dataset['input_npz'], allow_pickle=False) as inputs:
        for record in report['records']:
            case = cases[record['original_float32_job_id']]
            if sha(ROOT / case['checkpoint']) != case['checkpoint_sha256'] or record['checkpoint_sha256'] != case['checkpoint_sha256']:
                raise ValueError('Original preserved checkpoint differs')
            if sha(ROOT / record['diagnostic_scores']) != record['diagnostic_scores_sha256']:
                raise ValueError('Full diagnostic score archive differs')
            with np.load(ROOT / record['diagnostic_scores'], allow_pickle=False) as saved:
                _, lengths, _ = check_arrays(inputs, saved, dataset)
            if len(lengths) != record['full_registered_entities'] or sum(lengths.values()) != record['full_registered_test_points']:
                raise ValueError('Full registered coverage differs')
            expected_entities = {e['entity_id']: e for e in dataset['entities']}
            if {e['entity'] for e in record['entity_coverage']} != set(expected_entities):
                raise ValueError('Diagnostic entity coverage differs')
            nonfinite = 0
            for entity in record['entity_coverage']:
                spec = expected_entities[entity['entity']]
                for split, part in entity['splits'].items():
                    points = len(inputs[spec['array_prefix'] + '_' + split])
                    if part['full_points'] != points or part['float64_nonfinite_windows'] != 0:
                        raise ValueError('Full precision coverage differs')
                    nonfinite += part['float32_nonfinite_windows']
            if nonfinite != record['float32_nonfinite_windows']:
                raise ValueError('Full nonfinite counts differ')
            proofs.append(dict(seed=case['job']['seed'], full_entities=len(lengths), full_test_points=sum(lengths.values()),
                every_original_train_validation_test_score_finite=True, original_labels_exact=True,
                original_checkpoint_sha256=case['checkpoint_sha256'], diagnostic_scores_sha256=record['diagnostic_scores_sha256'],
                producer_reported_float32_nonfinite_windows=nonfinite))
    output = target / 'completed_full_checkpoint_diagnostic'
    for source in [report_path, state_path]:
        output.mkdir(parents=True, exist_ok=True)
        destination = output / source.name
        if destination.exists():
            raise FileExistsError('Preserve previous diagnostic proof')
        shutil.copyfile(source, destination)
    write(output / 'independent_verification.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        source_configuration=config_path.relative_to(ROOT).as_posix(), configuration_sha256=sha(config_path),
        original_prepared_input_sha256=dataset['input_sha256'], source_precision_report_sha256=sha(report_path),
        terminal_actual_controller_status_sha256=sha(state_path), full_diagnostics_verified=5, cases=proofs,
        original_full_precision_producer_source_pins_verified=True, independently_retrained_or_reinferred=False,
        new_from_scratch_float64_training_completions=0, benchmark_result=False,
        corrected_preprocessing_recomparison_still_required=True, all_original_experiments_complete=False))

    queue = json.loads((ROOT / 'configs/experiments/fm_strict_baseline_queue.v1.json').read_text('utf-8'))
    job = next(j for j in queue['jobs'] if j['id'] == 'fm_strict_anomaly_transformer__smd__registered__seed1105')
    if not complete_result(ROOT, job):
        raise ValueError('Original full SMD seed1105 incomplete')
    for source in job['frozen_files']:
        if sha(ROOT / source['path']) != source['sha256']:
            raise ValueError('Original full SMD frozen source differs')
    result_path = ROOT / job['output_directory'] / 'result.json'
    result = json.loads(result_path.read_text('utf-8'))
    dataset = next(d for d in manifest['datasets'] if d['id'] == 'smd')
    with np.load(ROOT / dataset['input_npz'], allow_pickle=False) as inputs, np.load(result_path.parent / 'scores.npz', allow_pickle=False) as saved:
        entities, lengths, validation = check_arrays(inputs, saved, dataset)
    for ratio, expected in result['metrics']['strict'].items():
        frozen = calibrate_validation(validation, false_alarm_percent=float(ratio), calibration_split='validation',
            calibration_dataset_sha256=dataset['input_sha256'], model_artifact_sha256=result['checkpoint_sha256'])
        actual = evaluate_entities(entities, expected_lengths=lengths, calibration=frozen)
        if any(expected[k] != value for k, value in actual.items()):
            raise ValueError('Original full saved SMD score metrics differ')
    resource_path = ROOT / 'experiments/runs/fm_bounded_strict_baseline_cpu_v2/strict_cpu/receipts' / (job['id'] + '.json')
    resource = json.loads(resource_path.read_text('utf-8'))
    if resource['exit_code'] != 0 or resource['resource_guard_aborted'] or not resource['full_registered_budget_retained']:
        raise ValueError('Original full SMD execution resource guard failed')
    output = target / 'new_completed_smd_seed1105'
    output.mkdir(exist_ok=True)
    shutil.copyfile(result_path, output / 'result.json')
    shutil.copyfile(resource_path, output / 'resource_receipt.json')
    write(output / 'independent_verification.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(), job_id=job['id'],
        full_registered_entities=len(lengths), full_registered_test_points=sum(lengths.values()),
        full_original_timestamp_scores_finite=True, labels_equal_original=True, all_three_calibrations_and_core_metrics_exact=True,
        result_sha256=sha(result_path), checkpoint_sha256=result['checkpoint_sha256'], scores_sha256=result['scores_sha256'],
        resource_receipt_sha256=sha(resource_path), resource_guard_aborted=False, original_registered_budget_retained=True,
        original_model_checkpoint_reinference_done=False, primary_metrics=result['metrics']['strict']['1']['micro'],
        author_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False))
    print(json.dumps(dict(full_precision_diagnostics_independently_verified=5, new_original_SMD_seed1105_verified=True)))


if __name__ == '__main__':
    main()
