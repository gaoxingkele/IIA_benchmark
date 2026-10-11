"""Verify full frozen MOMENT/SWAN evaluation in its actual producer environment."""
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from scripts.flow_matching.run_tsad_queue import complete_result
from scripts.flow_matching.prepare_tsad_execution import sha
from scripts.flow_matching.verify_tranad_precision_and_smd_completion_v1 import check_arrays
from iia_benchmark.evaluation.mtsad_strict import calibrate_validation, evaluate_entities


def main():
    queue_path = ROOT / 'configs/experiments/fm_moment_pretrained_queue.v1.json'
    queue = json.loads(queue_path.read_text('utf-8'))
    job = next(j for j in queue['jobs'] if j['id'] == 'fm_moment_pretrained_base__release512__swan__seed42')
    if not complete_result(ROOT, job):
        raise ValueError('Original complete MOMENT evaluation required')
    for source in job['frozen_files']:
        if sha(ROOT / source['path']) != source['sha256']:
            raise ValueError('Original pretrained asset/source differs')
    source = ROOT / job['output_directory'] / 'result.json'
    result = json.loads(source.read_text('utf-8'))
    if platform.python_version() != result['environment']['python'] or np.__version__ != result['environment']['numpy']:
        raise ValueError('Exact recomputation requires the actual producer Python/NumPy environment')
    manifest = json.loads((ROOT / job['dataset_manifest_path']).read_text('utf-8'))
    dataset = next(d for d in manifest['datasets'] if d['id'] == 'swan')
    if sha(ROOT / dataset['input_npz']) != dataset['input_sha256']:
        raise ValueError('Original full input differs')
    with np.load(ROOT / dataset['input_npz']) as inputs, np.load(source.parent / 'scores.npz') as saved:
        entities, lengths, validation = check_arrays(inputs, saved, dataset)
    for ratio, previous in result['metrics']['strict'].items():
        calibration = calibrate_validation(validation, false_alarm_percent=float(ratio), calibration_split='validation',
            calibration_dataset_sha256=dataset['input_sha256'], model_artifact_sha256=result['checkpoint_sha256'])
        actual = evaluate_entities(entities, expected_lengths=lengths, calibration=calibration)
        if any(previous[key] != value for key, value in actual.items()):
            raise ValueError('Full producer-environment saved metric recomputation differs')
    resource = ROOT / 'experiments/runs/fm_bounded_remaining_cpu_v3/moment_cpu/receipts' / (job['id'] + '.json')
    state = json.loads(resource.read_text('utf-8'))
    if state['exit_code'] != 0 or state['resource_guard_aborted'] or not state['full_registered_budget_retained']:
        raise ValueError('Actual original evaluation did not finish successfully')
    output = ROOT / 'projects/flow_matching_research/results/2026-10-11/constant_fit_preprocessing_v2/new_completed_moment_base_swan'
    output.mkdir(exist_ok=False)
    shutil.copyfile(source, output / 'result.json')
    shutil.copyfile(resource, output / 'resource_receipt.json')
    proof = dict(captured_utc=datetime.now(timezone.utc).isoformat(), canonical_job_id=job['id'],
        full_original_entities=len(lengths), full_original_test_points=sum(lengths.values()),
        all_frozen_sources_and_pretrained_assets_verified=True, full_scores_and_original_labels_verified=True,
        all_three_thresholds_and_core_metrics_recomputed_exact=True, primary_metrics=result['metrics']['strict']['1']['micro'],
        training_seconds=0, score_seconds=result['score_seconds'], checkpoint_sha256=result['checkpoint_sha256'],
        scores_sha256=result['scores_sha256'], resource_receipt_sha256=sha(resource),
        independent_recomputation_environment=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__),
        preliminary_host_environment_AP_absolute_difference=1.1102230246251565e-16,
        preliminary_host_environment=dict(python='3.12.10', numpy='2.4.6', sklearn='1.9.0'),
        fixed_pretrained_weight_evaluation_not_new_training=True, independently_reinferred_model=False,
        author_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False)
    (output / 'independent_verification.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(full_MOMENT_base_SWAN_evaluation_verified=True, full_test_points=sum(lengths.values()),
        validation_only_no_PA_F1=result['metrics']['strict']['1']['micro']['f1'], original_environment_exact=True)))


if __name__ == '__main__':
    main()
