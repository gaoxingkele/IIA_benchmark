"""Create a fresh, immutable original-budget/local-choice image training cohort."""
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read, sha


def create(relative, value):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
    return relative


def receipt(path):
    return dict(path=path, sha256=sha(ROOT / path))


def main():
    output = 'projects/flow_matching_research/results/2026-10-11/flow_mismatching_full_training_registration_v1'
    capacity_path = output + '/full_architecture_and_environment_capacity.json'
    capacity = read(ROOT / capacity_path)
    if not capacity['cuda_available']:
        raise ValueError('Frozen environment must have CUDA support')
    protocol_path = 'configs/experiments/fm_flow_mismatching_original_protocol.v1.json'
    optimizer_path = 'configs/models/fm_flow_mismatching_stableadamw_dinomaly_reference.v1.json'
    source_root = 'experiments/runs/mtsad_protocol_audit/sources/flow_matching_library/original/examples/image/models'
    common = [protocol_path, optimizer_path, capacity_path,
        'src/iia_benchmark/models/fm_supplemental_author.py',
        'src/iia_benchmark/models/fm_supplemental.py',
        'src/iia_benchmark/models/fm_optimizer_reference.py',
        'scripts/flow_matching/flow_mismatching_full_training_contract_v1.py',
        'scripts/flow_matching/run_flow_mismatching_full_training_v1.py',
        source_root + '/unet.py', source_root + '/nn.py']
    common += [read(ROOT / optimizer_path)['parameters']['source_path']]
    paper = read(ROOT / protocol_path)['paper']
    if sha(ROOT / paper['path']) != paper['sha256']:
        raise ValueError('Pinned original paper bytes differ')
    common += [paper['path']]
    full_choices = [
        'Five local uncertainty seeds 1103-1107; original seeds unknown.',
        'Single CUDA GPU/world_size1 without DDP; author world size and batch interpretation unknown.',
        'Batch32 is locally interpreted as global batch; full tail retained.',
        'CPU-generator random permutation per epoch; direct mmap batches, no workers or pinned-memory loader.',
        'No scheduler/EMA/early stopping; author did not report these settings.',
        'CuDNN benchmarkFalse/deterministicTrue; full CUDA bitwise reproducibility not certified.',
        'Per-epoch resume captures model/optimizer/Python/NumPy/Torch/CUDA/order RNG; interrupted epoch replays.',
        'Two rolling epoch checkpoints; immutable checkpoints every100epochs retained for epoch ablation.',
        'Related native Dinomaly StableAdamW AMSGrad semantics, not certified target-author implementation.',
        'Related pinned Meta U-Net with unit-time input, not target-author source release.',
        'Original image preprocessing gaps filled by already-frozen local full-data preprocessing choices.'
    ]
    datasets, models, created = {}, {}, []
    for name in ('mvtec', 'visa'):
        old_path = 'configs/datasets/fm_flow_mismatching_' + name + '_original.v1.json'
        old = read(ROOT / old_path)
        prep_path = 'configs/preprocessing/fm_flow_mismatching_' + name + '_original_images.v1.json'
        prep = read(ROOT / prep_path)
        verification_path = ('projects/flow_matching_research/results/2026-10-11/flow_mismatching_'
            + name + '_full_derived_verification_v1/independent_full_original_image_preprocessing_verification.json')
        verification = read(ROOT / verification_path)
        if not verification['passed'] or not verification['all_eight_array_hashes_and_shapes_verified']:
            raise ValueError('Independent full original preprocessing proof missing')
        derived_path = prep['derived_directory'] + '/full_derived_image_preprocessing.json'
        if sha(ROOT / derived_path) != verification['derived_report_sha256']:
            raise ValueError('Independent derived report binding differs')
        dataset_path = 'configs/datasets/fm_flow_mismatching_' + name + '_original.v2.json'
        dataset = dict(old)
        dataset.update(schema_version=2, id='fm_flow_mismatching_' + name + '_original_v2',
            supersedes=old_path, complete_raw_data_verified=True, complete_derived_data_verified=True,
            original_training_recipe_registered=True,
            expected_native_splits=verification['native_splits'],
            preprocessing_configuration=prep_path,
            original_preprocessing_equivalence_certified=False,
            array_receipts=verification['artifact_receipts'],
            source_receipts=[receipt(x) for x in [old_path, prep_path, verification_path,
                prep['full_raw_audit'], derived_path]],
            benchmark_ready=False,
            boundary='Complete independent original raw/derived verification. Local unreported preprocessing choices; full training/scoring pending.')
        created.append(create(dataset_path, dataset)); datasets[name] = dataset_path
        for variant in ('conditional', 'pooled'):
            model_path = 'configs/models/fm_flow_mismatching_' + name + '_' + variant + '_full_training.v1.json'
            model = read(ROOT / ('configs/models/fm_flow_mismatching_' + name + '_original.v1.json'))
            model.update(id='fm_flow_mismatching_' + name + '_' + variant + '_full_training_v1',
                dataset_configuration=dataset_path, variant=variant,
                reproduction_status='full_2000_epoch_local_choice_training_registered',
                original_author_equivalence_certified=False,
                all_paper_ablations_complete=False, local_unreported_choices=full_choices)
            model['parameters']['num_classes'] = dataset['expected_original_category_count'] if variant == 'conditional' else None
            created.append(create(model_path, model)); models[(name, variant)] = model_path
    jobs = []
    training = dict(epochs=2000, batch_size=32, gradient_accumulation=1,
        mixed_precision='bfloat16', TF32=True, early_stopping=False, scheduler=None,
        EMA=False, drop_last=False, shuffle=True, distributed_data_parallel=False,
        world_size=1, cpu_threads=2, checkpoint_interval=100)
    for variant in ('conditional', 'pooled'):
        for seed in range(1103, 1108):
            for name in ('mvtec', 'visa'):
                job_id = 'flow_mismatching_full__' + name + '__' + variant + '__seed' + str(seed)
                job_path = 'configs/experiments/fm_flow_mismatching_full_training_jobs/' + job_id + '.json'
                n = read(ROOT / datasets[name])['expected_native_splits']['train']
                job = dict(schema_version=1, id=job_id, dataset=name, variant=variant, seed=seed,
                    citation='https://arxiv.org/abs/2605.23070',
                    dataset_configuration=datasets[name], model_configuration=models[(name, variant)],
                    optimizer_configuration=optimizer_path, original_protocol=protocol_path,
                    environment_configuration=capacity_path,
                    output_directory='data/public_datasets/flow_matching/experiments/flow_mismatching_full_training_v1/jobs/' + job_id,
                    training=training, expected_training_images_per_epoch=n,
                    expected_optimizer_steps=2000 * ((n + 31) // 32),
                    original_author_equivalence_certified=False, benchmark_performance=False,
                    local_unreported_choices=full_choices,
                    source_receipts=[receipt(x) for x in common + [datasets[name], models[(name, variant)]]])
                created.append(create(job_path, job)); jobs.append(receipt(job_path))
    queue_path = 'configs/experiments/fm_flow_mismatching_full_training_queue.v1.json'
    created.append(create(queue_path, dict(schema_version=1, jobs=jobs, source_receipts=jobs,
        original_scope_preserved=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
        registered_new_training_jobs=20, seeds=list(range(1103, 1108)),
        remaining_original_experiments=['All original test image/pixel scores and per-category reporting',
            'Epoch/score/efficiency/Reflect/TCCM/WT/ReconFlow/native guidance and remaining paper obligations',
            'All existing full-budget time-series/baseline/replay queues retained'],
        uncertainty='Per-dataset/variant five-seed mean, sample SD and Student-t95% CI only after full independent evaluation.',
        boundary='Full-budget original dataset replication with explicitly unreported local choices; registration is not measured performance.')))
    runtime_path = 'configs/runtime/fm_flow_mismatching_full_training_queue.v1.json'
    runtime_sources = [queue_path, 'scripts/flow_matching/run_flow_mismatching_full_training_queue_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/run_light_controller.py', 'scripts/flow_matching/run_tsad_queue.py',
        'scripts/flow_matching/heavy_resources.py', 'scripts/flow_matching/heavy_scheduler_v2.py',
        'scripts/flow_matching/run_grasp_protocol_queue.py',
        'scripts/flow_matching/run_resource_gated_light_controller_v1.py',
        'scripts/flow_matching/flow_mismatching_full_training_contract_v1.py',
        'configs/runtime/fm_saits_guarded_original_resume.v1.json']
    created.append(create(runtime_path, dict(schema_version=1, queue=queue_path,
        python=capacity['python_executable'],
        environment_configuration=capacity_path,
        full_cohort_required_free_disk_bytes=capacity['required_free_disk_bytes'],
        resource_lock='experiments/runs/fm_grasp_protocol_gpu.lock',
        lane_lock='experiments/runs/fm_flow_mismatching_full_training_v1/queue.lock',
        state_root='experiments/runs/fm_flow_mismatching_full_training_v1',
        settings=read(ROOT / 'configs/runtime/fm_saits_guarded_original_resume.v1.json')['settings'],
        source_receipts=[receipt(x) for x in runtime_sources])))
    created.append(create(output + '/registration.json', dict(
        captured_utc=datetime.now(timezone.utc).isoformat(), full_training_jobs=20,
        original_scope=dict(papers=45, methods_baselines_ablations=681, original_data_tasks=273),
        queue=receipt(queue_path), runtime=receipt(runtime_path), created_files=created.copy(),
        original_author_equivalence_certified=False, benchmark_performance=False,
        boundary='All original and supplementary obligations preserved; this cohort adds full-budget image main/conditioning-ablation training.')))
    print(json.dumps(dict(registered_jobs=20, created_files=created)))


if __name__ == '__main__':
    main()
