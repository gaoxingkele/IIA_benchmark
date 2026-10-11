"""Frozen same-backbone TCCM comparison: complete original MVTec and50epochs."""
from datetime import datetime,timezone
import json

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT,read,sha


def create(relative,value):
    path=ROOT/relative;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(value,indent=2)+'\n')
    return relative


def receipt(path):return dict(path=path,sha256=sha(ROOT/path))


def main():
    protocol_path='configs/experiments/fm_flow_mismatching_original_protocol.v1.json'
    protocol=read(ROOT/protocol_path)
    if protocol['training_exceptions']['TCCM']['epochs']!=50:raise ValueError('Original50epoch exception required')
    dataset_path='configs/datasets/fm_flow_mismatching_mvtec_original.v2.json'
    optimizer_path='configs/models/fm_flow_mismatching_stableadamw_dinomaly_reference.v1.json'
    environment_path='projects/flow_matching_research/results/2026-10-11/flow_mismatching_full_training_registration_v1/full_architecture_and_environment_capacity.json'
    parent_model='configs/models/fm_flow_mismatching_mvtec_conditional_full_training.v1.json'
    model=read(ROOT/parent_model)
    model.update(id='fm_tccm_mvtec_controlled_image_domain_v1',method='TCCM_controlled_image_domain',
        original_protocol=protocol_path,citation='https://arxiv.org/abs/2605.23070',evidence_pages=[9,21,22],
        functions=dict(objective='iia_benchmark.models.fm_tccm_image:tccm_uniform_objective',
            score='iia_benchmark.models.fm_tccm_image:score_tccm_image'),
        score=dict(test_time=1,heatmap='unsquared channel L2 contraction residual',
            image_score='max plus mean of ceil top1percent pixels',top_fraction=.01),
        original_training_epochs=50,test_time=1,
        reproduction_status='paper_image_domain_controlled_comparison_full50epoch_training_registered',
        original_TCCM_best_domain_setting_reproduced=False,
        benchmark_ready=False,full_original_training_executed=False,
        boundary='Flow Mismatching paper same-backbone image-domain TCCM adaptation; not original TCCM best-domain results.')
    model['local_unreported_choices']+=['Same category-conditional backbone follows default class-label availability in AppendixC1/C2.',
        'All shared single-GPU, complete-tail, optimizer implementation and preprocessing gaps remain.',
        'Ten-epoch immutable checkpoint interval is a local resume/audit choice, not paper checkpoint selection.']
    model_path='configs/models/fm_tccm_mvtec_controlled_image_domain.v1.json'
    created=[create(model_path,model)]
    common=[protocol_path,protocol['paper']['path'],dataset_path,optimizer_path,environment_path,model_path,parent_model,
        'configs/models/fm_flow_mismatching_appendix_c_comparisons.v2.json',
        'src/iia_benchmark/models/fm_tccm_image.py','src/iia_benchmark/models/fm_image_comparison.py',
        'src/iia_benchmark/models/fm_supplemental_author.py','src/iia_benchmark/models/fm_optimizer_reference.py',
        'scripts/flow_matching/run_flow_mismatching_full_training_v1.py',
        'scripts/flow_matching/flow_mismatching_full_training_contract_v1.py',
        'scripts/flow_matching/tccm_full_image_training_contract_v1.py',
        'scripts/flow_matching/run_tccm_full_image_training_v1.py',
        'tests/test_fm_tccm_full_image_training_v1.py',
        model['parameters']['source_root']+'/unet.py',model['parameters']['source_root']+'/nn.py',
        read(ROOT/optimizer_path)['parameters']['source_path']]
    training= dict(epochs=50,batch_size=32,gradient_accumulation=1,mixed_precision='bfloat16',TF32=True,
        early_stopping=False,scheduler=None,EMA=False,drop_last=False,shuffle=True,
        distributed_data_parallel=False,world_size=1,cpu_threads=2,checkpoint_interval=10)
    jobs=[]
    for seed in range(1103,1108):
        job_id='tccm_controlled_image__mvtec__full50epoch__seed%d'%seed
        path='configs/experiments/fm_tccm_full_image_training_jobs/'+job_id+'.json'
        job=dict(schema_version=1,id=job_id,method='TCCM_controlled_image_domain',dataset='mvtec',variant='conditional',seed=seed,
            citation='https://arxiv.org/abs/2605.23070',original_protocol=protocol_path,evidence_pages=[9,21,22],
            dataset_configuration=dataset_path,model_configuration=model_path,optimizer_configuration=optimizer_path,
            environment_configuration=environment_path,
            output_directory='data/public_datasets/flow_matching/experiments/tccm_full_image_training_v1/jobs/'+job_id,
            training=training,expected_training_images_per_epoch=3629,expected_optimizer_steps=5700,
            local_unreported_choices=model['local_unreported_choices'],
            original_author_equivalence_certified=False,original_TCCM_best_domain_setting_reproduced=False,
            benchmark_performance=False,source_receipts=[receipt(x) for x in common])
        created.append(create(path,job));jobs.append(receipt(path))
    queue_path='configs/experiments/fm_tccm_full_image_training_queue.v1.json'
    scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273)
    created.append(create(queue_path,dict(schema_version=1,jobs=jobs,source_receipts=jobs,
        original_scope=scope,full_training_jobs=5,original_epochs=50,
        remaining_original_methods=['Reflect','WT-Flow','D-Flow','ReconFlow','All other original paper and supplemental obligations'],
        original_author_equivalence_certified=False,all_original_experiments_complete=False)))
    runtime_path='configs/runtime/fm_tccm_full_image_training_queue.v1.json'
    parent=read(ROOT/'configs/runtime/fm_flow_mismatching_full_training_queue.v1.json')
    capacity=read(ROOT/environment_path)
    full_size=next(x['conservative_full_checkpoint_bytes'] for x in capacity['full_architectures']
        if x['dataset']=='mvtec' and x['variant']=='conditional')
    runtime_sources=[queue_path,'scripts/flow_matching/run_tccm_full_image_training_queue_v1.py',
        'scripts/flow_matching/tccm_full_image_training_contract_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py','scripts/flow_matching/run_light_controller.py',
        'scripts/flow_matching/run_tsad_queue.py','scripts/flow_matching/heavy_resources.py',
        'scripts/flow_matching/heavy_scheduler_v2.py','scripts/flow_matching/run_grasp_protocol_queue.py',
        'scripts/flow_matching/run_resource_gated_light_controller_v1.py',
        'configs/runtime/fm_flow_mismatching_full_training_queue.v1.json']
    created.append(create(runtime_path,dict(schema_version=1,queue=queue_path,
        python=capacity['python_executable'],resource_lock=parent['resource_lock'],settings=parent['settings'],
        lane_lock='experiments/runs/fm_tccm_full_image_training_v1/queue.lock',
        state_root='experiments/runs/fm_tccm_full_image_training_v1',
        full_cohort_required_free_disk_bytes=full_size*5*7+8*1024**3,
        source_receipts=[receipt(x) for x in runtime_sources])))
    evidence='projects/flow_matching_research/results/2026-10-11/tccm_full_image_training_registration_v1'
    created.append(create(evidence+'/registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        original_scope=scope,queue=receipt(queue_path),runtime=receipt(runtime_path),
        full_original_training_jobs=5,original_epochs_per_job=50,normal_images_per_epoch=3629,
        optimizer_updates_per_job=5700,model_parameters=45289539,local_uncertainty_seeds=list(range(1103,1108)),
        created_files=created.copy(),original_author_equivalence_certified=False,
        original_TCCM_best_domain_setting_reproduced=False,full_training_jobs_completed=0,
        native_benchmark_performance=False,all_original_experiments_complete=False)))
    print(dict(full_original_tccm_image_jobs=5,epochs=50,images_per_epoch=3629,optimizer_updates=5700))


if __name__=='__main__':main()
