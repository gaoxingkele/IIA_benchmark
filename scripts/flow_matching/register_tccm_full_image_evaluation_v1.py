"""Register the seven original TCCM controlled-image targets, five full seeds."""
from datetime import datetime, timezone
import argparse
from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read
from scripts.flow_matching.register_flow_mismatching_full_image_evaluation_v1 import create, receipt
from scripts.flow_matching.tccm_full_image_evaluation_contract_v1 import SCORE_PARAMETERS, validate_queue


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',type=int,default=1)
    version=parser.parse_args().version
    if version not in (1,2):
        raise ValueError('Explicit registration version1 or2 required')
    protocol_path='configs/experiments/fm_flow_mismatching_original_protocol.v1.json'
    protocol=read(ROOT/protocol_path)
    index,row=next((i,row) for i,row in enumerate(protocol['Table2_same_backbone_FM_comparison']) if row['method']=='TCCM')
    training_queue_path='configs/experiments/fm_tccm_full_image_training_queue.v1.json'
    training_queue=read(ROOT/training_queue_path)
    cases=[]
    for entry in training_queue['jobs']:
        job=read(ROOT/entry['path']); case_id='table2_tccm__mvtec__full50epoch__seed%d'%job['seed']
        cases.append(dict(id=case_id,paper_table_key='Table2_same_backbone_FM_comparison',
            paper_row_index=index,dataset='mvtec',original_dataset_name='MVTec-AD',variant='conditional',
            seed=job['seed'],paper_reported=row['paper_reported'],score_parameters=SCORE_PARAMETERS,
            training_configuration=entry['path'],checkpoint_epoch=50,
            checkpoint=job['output_directory']+'/epoch_0050.pt',
            output_directory='data/public_datasets/flow_matching/experiments/tccm_full_image_evaluation_v1/cases/'+case_id))
    frozen=read(ROOT/'configs/experiments/fm_flow_mismatching_full_image_evaluation_queue.v1.json')
    sources=[protocol_path,protocol['paper']['path'],training_queue_path,frozen['environment_configuration'],
        'scripts/flow_matching/tccm_full_image_evaluation_contract_v1.py',
        'scripts/flow_matching/tccm_full_image_training_contract_v1.py',
        'scripts/flow_matching/flow_mismatching_full_training_contract_v1.py',
        'scripts/flow_matching/run_flow_mismatching_full_image_evaluation_v1.py',
        'scripts/flow_matching/flow_mismatching_image_metrics_v1.py',
        'scripts/flow_matching/run_tccm_full_image_evaluation_v1.py',
        'src/iia_benchmark/models/fm_tccm_image.py',
        'src/iia_benchmark/models/fm_image_comparison.py',
        'src/iia_benchmark/models/fm_supplemental_author.py',
        'tests/test_fm_tccm_full_image_evaluation_v1.py']+[x['path'] for x in training_queue['jobs']]
    local=[
        'This is the50epoch TCCM image-domain controlled adaptation in Flow Mismatching AppendixC2; not original TCCM best-domain settings.',
        'Complete native MVTec1725 test images and15 categories; unweighted category macro mean.',
        'Fixed t1 unsquared channel L2(v(y,1)+y); no Gaussian paths, probe streams or time integration.',
        'Image scalar max heatmap plus mean top ceil1percent pixels follows shared local controlled-image protocol; target TCCM scalar source uncertified.',
        'AUPRO native128 masks,8connected regions,equal-region weight,FPR.3 exact score groups and linear cutoff interpolation; target code unknown.',
        'Image/pixel F1max is test-label oracle reporting; not deployable calibration or strict TAB.',
        'FP32 evaluation parameters/inputs with TF32; author inference precision unknown.',
        'Separate every-image strict full-state model replay uses direct Eq20 and sorted scalar reduction, independent sklearn curves and CDF AUPRO.',
        'Full training/preprocessing/optimizer/U-Net gaps remain as frozen in five registered50epoch jobs.',
        'Author seeds unreported; five local uncertainty seeds1103-1107 retained.',
        'Immutable TCCM training snapshots are every10epochs; inherited generic100epoch description in prior frozen job notes does not override explicit training.checkpoint_interval10.']
    queue=dict(schema_version=1,original_protocol=protocol_path,
        environment_configuration=frozen['environment_configuration'],cases=cases,original_scope=training_queue['original_scope'],
        metrics=frozen['metrics'],verification=frozen['verification'],local_unreported_settings=local,
        source_receipts=[receipt(x) for x in sources],original_author_equivalence_certified=False,
        original_TCCM_best_domain_setting_reproduced=False,strict_TAB_result=False,all_original_experiments_complete=False)
    validate_queue(queue)
    queue_path='configs/experiments/fm_tccm_full_image_evaluation_queue.v%d.json'%version
    created=[create(queue_path,queue)]
    prior_runtime=read(ROOT/'configs/runtime/fm_tccm_full_image_training_queue.v1.json')
    runtime_sources=[queue_path,'scripts/flow_matching/run_tccm_full_image_evaluation_queue_v1.py',
        'scripts/flow_matching/tccm_full_image_evaluation_contract_v1.py',
        'scripts/flow_matching/tccm_full_image_training_contract_v1.py',
        'scripts/flow_matching/run_flow_mismatching_full_training_queue_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/run_light_controller.py','scripts/flow_matching/run_tsad_queue.py',
        'scripts/flow_matching/heavy_resources.py','scripts/flow_matching/heavy_scheduler_v2.py',
        'scripts/flow_matching/run_grasp_protocol_queue.py','scripts/flow_matching/run_resource_gated_light_controller_v1.py']
    runtime_path='configs/runtime/fm_tccm_full_image_evaluation_queue.v%d.json'%version
    created.append(create(runtime_path,dict(schema_version=1,queue=queue_path,python=prior_runtime['python'],
        resource_lock=prior_runtime['resource_lock'],settings=prior_runtime['settings'],
        lane_lock='experiments/runs/fm_tccm_full_image_evaluation_v1/queue.lock',
        state_root='experiments/runs/fm_tccm_full_image_evaluation_v1',source_receipts=[receipt(x) for x in runtime_sources])))
    target='projects/flow_matching_research/results/2026-10-11/tccm_full_image_evaluation_registration_v%d'%version
    created.append(create(target+'/registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        queue=receipt(queue_path),runtime=receipt(runtime_path),registered_full_test_seed_cases=5,
        original_paper_metric_targets=len(row['paper_reported']),full_test_images_per_seed=1725,
        full_original_categories_per_seed=15,original_training_epochs_per_seed=50,
        score_and_independent_replay_full_model_image_calls=17250,created_files=created.copy(),
        original_scope=training_queue['original_scope'],measured_native_score_results=0,
        native_benchmark_performance=False,all_original_experiments_complete=False,
        original_TCCM_best_domain_setting_reproduced=False)))
    print(dict(full_test_cases=5,full_images_per_seed=1725,metric_targets=len(row['paper_reported'])))


if __name__=='__main__':
    main()
