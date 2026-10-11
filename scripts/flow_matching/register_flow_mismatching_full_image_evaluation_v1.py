"""Freeze every transcribed original image table target and full native evaluator."""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read, sha


def create(relative, value):
    path = ROOT / relative; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
    return relative


def receipt(path):
    return dict(path=path, sha256=sha(ROOT / path))


def main():
    evidence = 'projects/flow_matching_research/results/2026-10-11/flow_mismatching_full_image_evaluation_registration_v1'
    protocol_path = 'configs/experiments/fm_flow_mismatching_original_protocol.v1.json'
    protocol = read(ROOT / protocol_path)
    capacity_path = 'projects/flow_matching_research/results/2026-10-11/flow_mismatching_full_training_registration_v1/full_architecture_and_environment_capacity.json'
    environment = read(ROOT / capacity_path)
    environment['packages'].update({p: importlib.metadata.version(p) for p in ('scipy', 'scikit-learn')})
    environment['parent_training_environment'] = receipt(capacity_path)
    environment['boundary'] = 'Frozen full native image scoring and independent metric-audit environment; no performance claim.'
    environment_path = 'configs/environments/fm_flow_mismatching_full_image_evaluation.v1.json'
    created = [create(environment_path, environment)]
    training_queue_path = 'configs/experiments/fm_flow_mismatching_full_training_queue.v1.json'
    training_entries = read(ROOT / training_queue_path)['jobs']
    training = {(read(ROOT / x['path'])['dataset'], read(ROOT / x['path'])['variant'], read(ROOT / x['path'])['seed']):
                (x['path'], read(ROOT / x['path'])) for x in training_entries}
    targets, cases, remaining = [], [], []
    for key in ('Table6_compute', 'Table7_path_aggregation', 'Table8_conditional_mean', 'Table9_epoch_checkpoints', 'Table2_same_backbone_FM_comparison'):
        for index, row in enumerate(protocol[key]):
            name = {'MVTec-AD': 'mvtec', 'VisA': 'visa'}[row['dataset']]
            variant = 'conditional' if row.get('category_conditioning', True) else 'pooled'
            eligible = key != 'Table2_same_backbone_FM_comparison' or row['method'] == 'Flow Mismatching'
            target_id = key.lower() + '__row%02d__' % index + name
            params = dict(paths=row.get('K', 10), times=row.get('T', 10),
                aggregation=row.get('aggregation', 'minimum'), percentile=row.get('percentile') or .1,
                weighted=True, top_fraction=.01)
            target = dict(id=target_id, original_dataset_name=row['dataset'], dataset=name,
                paper_table_key=key, paper_row_index=index, paper_reported=row['paper_reported'],
                paper_table=row['table'], paper_page=row['page'], required_training_seeds=5,
                full_score_runner_registered=eligible, variant=variant,
                checkpoint_epoch=row.get('checkpoint_epoch', 2000), score_parameters=params,
                method=row.get('method', 'Flow Mismatching'))
            if not eligible:
                target['status'] = 'original_comparator_training_or_native_adapter_settings_pending'
                target['required_training_epochs'] = 50 if row['method'] == 'TCCM' else 2000
                target['explicit_callable_or_status_configuration'] = 'configs/models/fm_flow_mismatching_appendix_c_comparisons.v2.json'
                remaining.append(target)
            targets.append(target)
            if eligible:
                for seed in range(1103, 1108):
                    path, job = training[(name, variant, seed)]
                    case_id = target_id + '__seed%d' % seed
                    cases.append(dict(id=case_id, target_id=target_id,
                        paper_table_key=key, paper_row_index=index, dataset=name,
                        original_dataset_name=row['dataset'], variant=variant, seed=seed,
                        paper_reported=row['paper_reported'], score_parameters=params,
                        training_configuration=path, checkpoint_epoch=target['checkpoint_epoch'],
                        checkpoint=job['output_directory'] + '/epoch_%04d.pt' % target['checkpoint_epoch'],
                        output_directory='data/public_datasets/flow_matching/experiments/flow_mismatching_full_image_evaluation_v1/cases/' + case_id))
    if sum(len(t['paper_reported']) for t in targets) != 520 or len(cases) != 325:
        raise ValueError('Complete original520 transcribed targets and325 runnable table/seed cases required')
    source_paths = [protocol_path, protocol['paper']['path'], training_queue_path, environment_path,
        'configs/models/fm_flow_mismatching_appendix_c_comparisons.v2.json',
        'scripts/flow_matching/flow_mismatching_full_training_contract_v1.py',
        'scripts/flow_matching/flow_mismatching_image_metrics_v1.py',
        'scripts/flow_matching/run_flow_mismatching_full_image_evaluation_v1.py',
        'scripts/flow_matching/audit_flow_mismatching_full_image_evaluation_v1.py',
        'src/iia_benchmark/models/fm_image_evaluation.py',
        'src/iia_benchmark/models/fm_supplemental.py',
        'src/iia_benchmark/models/fm_supplemental_author.py',
        'tests/test_fm_flow_mismatching_image_metrics_v1.py',
        'tests/test_fm_flow_mismatching_full_image_scoring_v1.py'] + [x['path'] for x in training_entries]
    local = [
        'AUPRO: native derived128 masks, 8-connected regions, equal-region weight, maxFPR0.3, exact unique-score groups and linear cutoff interpolation.',
        'AUPRO settings are conventional local choices; target author implementation unspecified.',
        'FP32 evaluation inputs/parameters with TF32; original inference precision unknown.',
        'Per-image/per-path SHA256 keyed CUDA seeds paired across K/T/aggregation/epoch/conditioning cases; probe prefix independent of case order.',
        'Percentile ceil(alpha*K) at alpha0.1/K10 equals minimum under paired noise; printed differing results remain untouched.',
        'Synchronized native forward/scorer timing excludes model loading, IO, probe draws, transfers and one full-image warmup.',
        'Local RTX3090 timing cannot certify original A10080GB FPS claims.',
        'Image/pixel F1-max uses full test labels only for reported oracle evaluation, never training/model choice/TAB calibration.',
        'All full original test arrays and native category order retained; unweighted category mean matches stated paper averaging.',
        'All local full training/preprocessing/optimizer/U-Net choices and author fidelity gaps from frozen training cohort remain.']
    queue_path = 'configs/experiments/fm_flow_mismatching_full_image_evaluation_queue.v1.json'
    created.append(create(queue_path, dict(schema_version=1, original_protocol=protocol_path,
        environment_configuration=environment_path, cases=cases, paper_targets=targets,
        remaining_original_comparator_targets=remaining,
        original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
        metrics=dict(aupro_fpr_limit=.3, region_connectivity=8,
            category_average='unweighted mean', image_score_top_fraction=.01, F1='test-label oracle threshold sweep'),
        verification=dict(model_replay_tolerance=dict(rtol=2e-6, atol=2e-6),
            heatmap_image_score_tolerance=dict(rtol=2e-6, atol=2e-6),
            metric_tolerance=dict(rtol=1e-9, atol=1e-9),
            tolerance_boundary='Preregistered before real checkpoint or score exists; never adapted to observed agreement.'),
        local_unreported_settings=local, source_receipts=[receipt(x) for x in source_paths],
        original_author_equivalence_certified=False, strict_TAB_result=False,
        all_original_experiments_complete=False,
        remaining_original_paper_obligations=protocol['remaining_table_obligations'])))
    runtime_path = 'configs/runtime/fm_flow_mismatching_full_image_evaluation_queue.v1.json'
    original_runtime = read(ROOT / 'configs/runtime/fm_flow_mismatching_full_training_queue.v1.json')
    runtime_sources = [queue_path, 'scripts/flow_matching/run_flow_mismatching_full_image_evaluation_queue_v1.py',
        'scripts/flow_matching/run_flow_mismatching_full_training_queue_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/run_light_controller.py', 'scripts/flow_matching/run_tsad_queue.py',
        'scripts/flow_matching/heavy_resources.py', 'scripts/flow_matching/heavy_scheduler_v2.py',
        'scripts/flow_matching/run_grasp_protocol_queue.py',
        'scripts/flow_matching/run_resource_gated_light_controller_v1.py']
    created.append(create(runtime_path, dict(schema_version=1, queue=queue_path,
        python=environment['python_executable'], resource_lock=original_runtime['resource_lock'],
        lane_lock='experiments/runs/fm_flow_mismatching_full_image_evaluation_v1/queue.lock',
        state_root='experiments/runs/fm_flow_mismatching_full_image_evaluation_v1',
        settings=original_runtime['settings'], source_receipts=[receipt(x) for x in runtime_sources])))
    created.append(create(evidence + '/registration.json', dict(
        captured_utc=datetime.now(timezone.utc).isoformat(), queue=receipt(queue_path), runtime=receipt(runtime_path),
        registered_native_case_seed_evaluations=325, complete_transcribed_original_metric_targets=520,
        runnable_full_model_score_targets=485, remaining_original_comparator_metric_targets=35,
        original_A100_FPS_targets=30, original_A100_hardware_equivalent=False,
        full_original_model_checkpoints_already_available=0, measured_native_score_results=0,
        created_files=created.copy(), original_scope=read(ROOT / queue_path)['original_scope'],
        all_original_experiments_complete=False, native_benchmark_performance=False,
        boundary='Every known transcribed table target retained; full scoring/replay of325 cases awaits actual original-budget checkpoints. Not all paper category/toy/qualitative/other-method obligations are transcribed or executed.')))
    print(dict(native_case_seed_evaluations=325, original_metric_targets=520,
        callable_score_targets=485, original_comparator_targets_remaining=35))


if __name__ == '__main__':
    main()
