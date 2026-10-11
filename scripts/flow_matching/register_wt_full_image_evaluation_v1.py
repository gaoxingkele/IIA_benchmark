"""Register all six native step recipes, five full2000epoch WT image models."""
from datetime import datetime, timezone
from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read, sha, verify_receipts
from scripts.flow_matching.register_flow_mismatching_full_image_evaluation_v1 import create, receipt
from scripts.flow_matching.wt_full_image_evaluation_contract_v1 import parameters, native_solver_facts, validate_queue


def main():
    protocol_path = 'configs/experiments/fm_flow_mismatching_original_protocol.v1.json'
    protocol = read(ROOT / protocol_path)
    index, row = next((i, r) for i, r in enumerate(protocol['Table2_same_backbone_FM_comparison']) if r['method'] == 'WT-Flow')
    training_path = 'configs/experiments/fm_wt_full_image_training_queue.v1.json'
    training = read(ROOT / training_path)
    verify_receipts(training['source_receipts'])
    material_path = 'projects/flow_matching_research/results/2026-10-11/flow_mismatching_reflect_wt_sources_v1/full_original_reflect_wt_material_audit.json'
    material = read(ROOT / material_path)
    native = next(r for r in material['repositories'] if r['repository'] == 'https://github.com/lil-wayne-0319/fmad.git')
    if native['commit'] != 'f621f2b251637605a7d25593251d3be89a8c9734' or native['all_original_git_files_verified'] != 20:
        raise ValueError('Previously verified complete native WT Git source required')
    native_sources = []
    for role, name in [('main', 'main.py'), ('infer_config', 'infer_config.py'), ('infer', 'infer.py')]:
        item = next(r for r in native['files'] if r['path'] == name)
        native_sources.append(dict(role=role, path=native['original_code_root'] + '/' + name, sha256=item['sha256']))
    facts = native_solver_facts(native_sources)
    first_job = read(ROOT / training['jobs'][0]['path'])
    model = read(ROOT / first_job['model_configuration'])
    evaluation_path = 'configs/models/fm_wt_native_step_image_evaluation.v1.json'
    evaluation = dict(schema_version=1, id='fm_wt_native_step_image_evaluation_v1',
        task='image_anomaly_detection', method='WT_Flow_controlled_image_domain',
        citation=model['citation'], data_citation='https://www.mvtec.com/company/research/datasets/mvtec-ad',
        related_repository=native['repository'], related_commit=native['commit'],
        functions=dict(score='iia_benchmark.models.fm_wt_image_evaluation:score_wt_image',
            independent_replay='scripts.flow_matching.run_wt_full_image_evaluation_v1:direct_replay_score'),
        training_model_configuration=first_job['model_configuration'], training_model_configuration_sha256=sha(ROOT / first_job['model_configuration']),
        dataset_configuration=first_job['dataset_configuration'], native_source=model['native_source'],
        native_solver_source_receipts=native_sources, solver_facts=facts,
        normalization=model['method_parameters'], target_heatmap=parameters(1)['energy'], top_fraction=.01,
        checkpoint_selection='Frozen final2000epoch only; all six source recipes retained without test-label selection.',
        original_target_solver_and_step_identity_certified=False, original_WT_best_feature_setting_reproduced=False,
        strict_TAB_result=False, benchmark_ready=False, reproduction_status='callable_full_native_test_score_and_replay_registered_pending_full_training',
        gaps=['Target WT image-domain integrator and step count are unreported.',
              'Related native feature generate_pdf uses negative half channel MEAN; this target adapter follows positive half channel SUM Eq25.',
              'Related native scan actively tests transistor only; complete15category scan is a local extension.',
              'Full-image preprocessing, backbone, optimizer, inference precision and native-versus-target feature-domain differences remain.',
              'Native feature best-det/best-loc checkpoint selection is a separate protocol, not reproduced by final2000epoch image checkpoints.'])
    created = [create(evaluation_path, evaluation)]
    cases = []
    for entry in training['jobs']:
        job = read(ROOT / entry['path'])
        for steps in facts['native_step_scan']:
            identifier = 'table2_wt_related_native_step%d__mvtec__full2000epoch__seed%d' % (steps, job['seed'])
            cases.append(dict(id=identifier, paper_table_key='Table2_same_backbone_FM_comparison',
                paper_row_index=index, dataset='mvtec', original_dataset_name='MVTec-AD', variant='conditional',
                seed=job['seed'], paper_reported=row['paper_reported'], score_parameters=parameters(steps),
                training_configuration=entry['path'], checkpoint_epoch=2000, checkpoint=job['output_directory'] + '/epoch_2000.pt',
                output_directory='data/public_datasets/flow_matching/experiments/wt_full_image_evaluation_v1/cases/' + identifier))
    parent_path = 'configs/experiments/fm_flow_mismatching_full_image_evaluation_queue.v1.json'
    parent = read(ROOT / parent_path)
    sources = [evaluation_path, material_path, protocol_path, protocol['paper']['path'], training_path, parent_path,
        parent['environment_configuration'], 'scripts/flow_matching/register_wt_full_image_evaluation_v1.py',
        'scripts/flow_matching/wt_full_image_evaluation_contract_v1.py',
        'scripts/flow_matching/wt_full_image_training_contract_v1.py',
        'scripts/flow_matching/flow_mismatching_full_training_contract_v1.py',
        'scripts/flow_matching/run_flow_mismatching_full_image_evaluation_v1.py',
        'scripts/flow_matching/flow_mismatching_image_metrics_v1.py',
        'scripts/flow_matching/run_tccm_full_image_evaluation_v1.py',
        'scripts/flow_matching/run_wt_full_image_evaluation_v1.py',
        'src/iia_benchmark/models/fm_wt_image_evaluation.py', 'src/iia_benchmark/models/fm_reflect_wt_native.py',
        'src/iia_benchmark/models/fm_supplemental_author.py', 'tests/test_fm_wt_full_image_evaluation_v1.py',
        first_job['model_configuration'], first_job['native_components_configuration']] + [x['path'] for x in training['jobs']]
    queue = dict(schema_version=1, original_protocol=protocol_path, evaluation_model_configuration=evaluation_path,
        parent_full_image_evaluation_queue=parent_path, environment_configuration=parent['environment_configuration'],
        cases=cases, solver_facts=facts, original_scope=training['original_scope'], metrics=parent['metrics'],
        verification=parent['verification'], source_receipts=[receipt(x) for x in sources] + native_sources,
        original_author_equivalence_certified=False, original_target_solver_and_step_identity_certified=False,
        original_WT_best_feature_setting_reproduced=False, strict_TAB_result=False, solver_selected_using_test_labels=False,
        all_original_experiments_complete=False,
        local_unreported_settings=[
            'Complete3629normal-train/1725test/15category controlled-image WT at2000epochs, five seeds1103-1107.',
            'Original related source scan1/10/20/50/100/200; default50training CLI and1inference; no test-selected winning step.',
            'Related native step_test actively scans transistor; this all15category extension is local.',
            'Related native left Euler j*(1/steps), allCHW LayerNorm epsilon1e-5 and population variance; target solver identity unknown.',
            'Positive half channel SUM endpoint energy per target Eq25, distinct from negative half channel MEAN in related feature code.',
            'Shared local image scalar=max+mean top ceil1percent, native128 masks/8connected/equal-region/FPR.3 AUPRO; target implementations unknown.',
            'Unweighted per-category macro AUROC/AP/F1max/AUPRO. F1max uses test-label oracle reporting, not deployment calibration or strict TAB.',
            'FP32 inference with TF32; original inference precision unreported.',
            'Every test image replayed from the strict full checkpoint with direct Euler/energy and sorted scalar; sklearn curves and independent CDF AUPRO.',
            'All seven paper comparisons are exploratory differences until target solver, preprocessing, backbone and optimizer alignment is certified.',
            'All original and supplemental obligations remain; related-source sensitivity is not a replacement for unknown target-original integration settings.'])
    validate_queue(queue)
    queue_path = 'configs/experiments/fm_wt_full_image_evaluation_queue.v1.json'
    created.append(create(queue_path, queue))
    prior_runtime = read(ROOT / 'configs/runtime/fm_wt_full_image_training_queue.v1.json')
    runtime_sources = [queue_path, 'scripts/flow_matching/run_wt_full_image_evaluation_queue_v1.py',
        'scripts/flow_matching/wt_full_image_evaluation_contract_v1.py', 'scripts/flow_matching/wt_full_image_training_contract_v1.py',
        'scripts/flow_matching/run_flow_mismatching_full_training_queue_v1.py', 'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/run_light_controller.py', 'scripts/flow_matching/run_tsad_queue.py',
        'scripts/flow_matching/heavy_resources.py', 'scripts/flow_matching/heavy_scheduler_v2.py',
        'scripts/flow_matching/run_grasp_protocol_queue.py', 'scripts/flow_matching/run_resource_gated_light_controller_v1.py']
    runtime_path = 'configs/runtime/fm_wt_full_image_evaluation_queue.v1.json'
    created.append(create(runtime_path, dict(schema_version=1, queue=queue_path, python=prior_runtime['python'],
        settings=prior_runtime['settings'], resource_lock=prior_runtime['resource_lock'],
        lane_lock='experiments/runs/fm_wt_full_image_evaluation_v1/queue.lock', state_root='experiments/runs/fm_wt_full_image_evaluation_v1',
        source_receipts=[receipt(x) for x in runtime_sources])))
    target = 'projects/flow_matching_research/results/2026-10-11/wt_full_image_evaluation_registration_v1'
    created.append(create(target + '/registration.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        queue=receipt(queue_path), runtime=receipt(runtime_path), solver_facts=facts,
        registered_full_test_seed_solver_cases=30, independent_original_training_seeds=5,
        original_paper_metric_targets=7, full_test_images_per_case=1725, full_categories_per_case=15,
        original_training_epochs_per_seed=2000, score_and_replay_full_test_image_visits=103500,
        score_and_replay_full_model_velocity_calls=2 * 1725 * 5 * sum(facts['native_step_scan']),
        created_files=created.copy(), original_scope=training['original_scope'], new_training_seed_slots=0,
        measured_native_score_results=0, native_benchmark_performance=False,
        original_target_solver_and_step_identity_certified=False, all_original_experiments_complete=False)))
    print(dict(full_test_seed_solver_cases=30, full_images_per_case=1725, native_step_grid=facts['native_step_scan']))


if __name__ == '__main__':
    main()
