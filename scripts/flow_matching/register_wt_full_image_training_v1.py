"""Complete original2000epoch same-backbone WT image-domain comparison cohort."""
from datetime import datetime,timezone
from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT,read,sha
from scripts.flow_matching.register_tccm_full_image_training_v1 import create,receipt
from scripts.flow_matching.wt_full_image_training_contract_v1 import WT_PARAMETERS,validate_wt_job


def main():
    protocol_path='configs/experiments/fm_flow_mismatching_original_protocol.v1.json'
    protocol=read(ROOT/protocol_path)
    row=next(row for row in protocol['Table2_same_backbone_FM_comparison'] if row['method']=='WT-Flow')
    if row['dataset']!='MVTec-AD':raise ValueError('Original controlled WT dataset required')
    native_config='configs/models/fm_reflect_wt_native_components.v1.json'
    native=read(ROOT/native_config)['wt']
    dataset_path='configs/datasets/fm_flow_mismatching_mvtec_original.v2.json'
    optimizer_path='configs/models/fm_flow_mismatching_stableadamw_dinomaly_reference.v1.json'
    environment_path='projects/flow_matching_research/results/2026-10-11/flow_mismatching_full_training_registration_v1/full_architecture_and_environment_capacity.json'
    parent_model='configs/models/fm_flow_mismatching_mvtec_conditional_full_training.v1.json'
    model=read(ROOT/parent_model)
    model.update(id='fm_wt_mvtec_controlled_image_domain_v1',method='WT_Flow_controlled_image_domain',
        original_protocol=protocol_path,evidence_pages=[9,21,22],
        functions=dict(objective='iia_benchmark.models.fm_wt_image_training:wt_uniform_objective',
            heatmap='iia_benchmark.models.fm_image_comparison:wt_flow_heatmap'),
        native_components_configuration=native_config,method_parameters=WT_PARAMETERS,
        native_source={k:native[k] for k in ('source_path','expected_sha256')},
        score=dict(heatmap='positive one-half channel SUM squared transported endpoint,Eq25',
            original_target_solver_and_step_identity_certified=False,
            score_and_full_test_replay_registration_status='pending_explicit_original_target_ODE_recipe_audit'),
        original_training_epochs=2000,reproduction_status='controlled_image_domain_full2000epoch_training_registered',
        original_WT_best_feature_setting_reproduced=False,benchmark_ready=False,full_original_training_executed=False,
        native_repository='https://github.com/lil-wayne-0319/fmad',
        native_repository_commit='f621f2b251637605a7d25593251d3be89a8c9734',
        boundary='Full WT image-domain adaptation in Flow Mismatching AppendixC2; related native reverse FM and explicit local LayerNorm choice. Not original WT feature-domain best settings.')
    model['local_unreported_choices']+=[
        'WT epsilon1e-5 and all-C/H/W population variance follow related native fresh LayerNorm; target-author WT epsilon/variance still unspecified.',
        'WT is applied after already-frozen complete normal-train RGB preprocessing; target normalization order is not certified.',
        'Uniform time drawn first, full independent Gaussian second; matches related native train/create_flow_rev order.',
        'Category conditional default follows available class labels and shared backbone in AppendixC1/C2.',
        'Full2000epoch image adapter follows target common budget; original WRN50 feature100epoch budget is a distinct experiment.',
        'Freeze final/100epoch checkpoints without test-metric model selection; original feature code best-det/best-loc selection remains a separately documented protocol.',
        'Related native Euler is not enough to certify target-author WT or ReconFlow integration settings.']
    model_path='configs/models/fm_wt_mvtec_controlled_image_domain.v1.json'
    created=[create(model_path,model)]
    common=[protocol_path,protocol['paper']['path'],native_config,model['native_source']['source_path'],
        dataset_path,optimizer_path,environment_path,model_path,parent_model,
        'src/iia_benchmark/models/fm_wt_image_training.py','src/iia_benchmark/models/fm_reflect_wt_native.py',
        'src/iia_benchmark/models/fm_supplemental_author.py','src/iia_benchmark/models/fm_optimizer_reference.py',
        'scripts/flow_matching/run_flow_mismatching_full_training_v1.py',
        'scripts/flow_matching/flow_mismatching_full_training_contract_v1.py',
        'scripts/flow_matching/wt_full_image_training_contract_v1.py',
        'scripts/flow_matching/run_wt_full_image_training_v1.py',
        'tests/test_fm_wt_full_image_training_v1.py',
        model['parameters']['source_root']+'/unet.py',model['parameters']['source_root']+'/nn.py',
        read(ROOT/optimizer_path)['parameters']['source_path']]
    parent_training=read(ROOT/'configs/experiments/fm_flow_mismatching_full_training_jobs/flow_mismatching_full__mvtec__conditional__seed1103.json')['training']
    jobs=[]
    for seed in range(1103,1108):
        job_id='wt_controlled_image__mvtec__full2000epoch__seed%d'%seed
        path='configs/experiments/fm_wt_full_image_training_jobs/'+job_id+'.json'
        job=dict(schema_version=1,id=job_id,method='WT_Flow_controlled_image_domain',dataset='mvtec',variant='conditional',seed=seed,
            citation='https://arxiv.org/abs/2605.23070',original_protocol=protocol_path,evidence_pages=[9,21,22],
            native_components_configuration=native_config,dataset_configuration=dataset_path,
            model_configuration=model_path,optimizer_configuration=optimizer_path,environment_configuration=environment_path,
            output_directory='data/public_datasets/flow_matching/experiments/wt_full_image_training_v1/jobs/'+job_id,
            training=parent_training,expected_training_images_per_epoch=3629,expected_optimizer_steps=228000,
            local_unreported_choices=model['local_unreported_choices'],original_author_equivalence_certified=False,
            original_WT_best_feature_setting_reproduced=False,benchmark_performance=False,
            source_receipts=[receipt(x) for x in common])
        validate_wt_job(job,model,read(ROOT/optimizer_path),read(ROOT/dataset_path))
        created.append(create(path,job));jobs.append(receipt(path))
    scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273)
    queue_path='configs/experiments/fm_wt_full_image_training_queue.v1.json'
    created.append(create(queue_path,dict(schema_version=1,jobs=jobs,source_receipts=jobs,
        original_scope=scope,full_training_jobs=5,original_epochs=2000,
        original_paper_reported=row['paper_reported'],
        remaining_original_obligations=['WT full original test score/replay with source solver audit',
            'Reflect','D-Flow','ReconFlow','All original45paper and supplemental experiment obligations'],
        original_author_equivalence_certified=False,original_WT_best_feature_setting_reproduced=False,
        all_original_experiments_complete=False)))
    capacity=read(ROOT/environment_path);parent=read(ROOT/'configs/runtime/fm_flow_mismatching_full_training_queue.v1.json')
    full_size=next(x['conservative_full_checkpoint_bytes'] for x in capacity['full_architectures']
        if x['dataset']=='mvtec' and x['variant']=='conditional')
    runtime_sources=[queue_path,'scripts/flow_matching/run_wt_full_image_training_queue_v1.py',
        'scripts/flow_matching/wt_full_image_training_contract_v1.py',
        'scripts/flow_matching/flow_mismatching_full_training_contract_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py','scripts/flow_matching/run_light_controller.py',
        'scripts/flow_matching/run_tsad_queue.py','scripts/flow_matching/heavy_resources.py',
        'scripts/flow_matching/heavy_scheduler_v2.py','scripts/flow_matching/run_grasp_protocol_queue.py',
        'scripts/flow_matching/run_resource_gated_light_controller_v1.py',
        'configs/runtime/fm_flow_mismatching_full_training_queue.v1.json']
    runtime_path='configs/runtime/fm_wt_full_image_training_queue.v1.json'
    created.append(create(runtime_path,dict(schema_version=1,queue=queue_path,python=capacity['python_executable'],
        resource_lock=parent['resource_lock'],settings=parent['settings'],
        lane_lock='experiments/runs/fm_wt_full_image_training_v1/queue.lock',
        state_root='experiments/runs/fm_wt_full_image_training_v1',
        full_cohort_required_free_disk_bytes=full_size*5*22+16*1024**3,
        source_receipts=[receipt(x) for x in runtime_sources])))
    target='projects/flow_matching_research/results/2026-10-11/wt_full_image_training_registration_v1'
    created.append(create(target+'/registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        original_scope=scope,queue=receipt(queue_path),runtime=receipt(runtime_path),
        full_original_training_jobs=5,original_epochs_per_job=2000,full_normal_images_per_epoch=3629,
        optimizer_updates_per_job=228000,model_parameters=45289539,local_uncertainty_seeds=list(range(1103,1108)),
        created_files=created.copy(),original_author_equivalence_certified=False,
        original_WT_best_feature_setting_reproduced=False,full_training_jobs_completed=0,
        native_benchmark_performance=False,all_original_experiments_complete=False)))
    print(dict(full_original_wt_image_jobs=5,epochs=2000,images_per_epoch=3629,optimizer_updates=228000))


if __name__=='__main__':main()
