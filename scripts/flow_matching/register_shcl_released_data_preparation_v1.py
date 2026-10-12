"""Freeze all five real input datasets and full source52 loader-mode obligations."""
import json

from scripts.flow_matching.prepare_shcl_released_data_v1 import native,data_environment
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    model='configs/models/fm_shcl_released_vae_native.v1.json'
    operator='configs/runtime/fm_shcl_released_operator_audit.v1.json'
    shared='F:/aicoding/IIA_Data/public_datasets/flow_matching/shcl_shared_data_retry_v1'
    acquisition=shared+'/acquisition_report.json'
    if native.read(acquisition)['complete_valid_csv_files']!=5:
        raise ValueError('All five repaired author shared-folder files required')
    datasets=[]
    for dataset in ('MSL','SMAP'):
        specification=native.read(native.ROOT/('configs/datasets/mtsad_'+dataset.lower()+'.json'))
        names={key:dataset+'_'+('test_label' if key=='label' else key)+'.npy' for key in ('train','test','label')}
        datasets.append(dict(dataset=dataset,format='author_npy',features=specification['features'],
            expected_shapes=specification['expected_shapes'],native_class=dataset+'SegLoader',output_filenames=names,
            inputs={key:dict(path='data/public_datasets/flow_matching/dcdetector/author_bundle/'+dataset+'/'+name,
                sha256=native.sha(native.ROOT/('data/public_datasets/flow_matching/dcdetector/author_bundle/'+dataset+'/'+name)))
                for key,name in names.items()},author_version_boundary='Actual SHCL-linked shared folder payload; '
                    'original paper-era per-entity grouping/version is still unverified.'))
    for dataset,kind,features,names,native_class in [
        ('PSM','PSM_timestamp_csv',25,dict(train='train.csv',test='test.csv',label='test_label.csv'),'PSMSegLoader'),
        ('SWaT','SWAT_numeric_label_csv',51,dict(train='SWaT_train.npy',test='SWaT_test.npy',label='SWaT_test_label.npy'),'SWaTSegLoader')]:
        originals=dict(train='train.csv',test='test.csv',label='test_label.csv') if dataset=='PSM' else dict(train='swat_train2.csv',test='swat2.csv')
        specification=native.read(native.ROOT/('configs/datasets/mtsad_'+dataset.lower()+'.json'))
        datasets.append(dict(dataset=dataset,format=kind,features=features,native_class=native_class,
            output_filenames=names,expected_shapes=specification['expected_shapes'],
            inputs={key:dict(path=shared+'/'+('PSM' if dataset=='PSM' else 'SWAT')+'/'+filename,
                sha256=native.sha(shared+'/'+('PSM' if dataset=='PSM' else 'SWAT')+'/'+filename)) for key,filename in originals.items()},
            author_version_boundary='Complete retried SHCL-linked shared-folder CSV. Required format adaptation '
                'is explicit: drop PSM timestamp; convert SWaT CSV to native NPY. Paper-era transformed files '
                'not supplied. SWaT475200-paper vs495000-released training discrepancy remains unresolved.'))
    machine='machine-3-11'; source='experiments/runs/fm_code_completion/sources/omnianomaly/original/ServerMachineDataset'
    datasets.append(dict(dataset=machine,format='OmniAnomaly_machine_txt',features=38,native_class='ServiceSegLoader',
        output_filenames={key:machine+'_'+('test_label' if key=='label' else key)+'.npy' for key in ('train','test','label')},
        path_prefix=machine,inputs={key:dict(path=source+'/'+('test_label' if key=='label' else key)+'/'+machine+'.txt',
            sha256=native.sha(native.ROOT/(source+'/'+('test_label' if key=='label' else key)+'/'+machine+'.txt')))
            for key in ('train','test','label')},author_version_boundary='Exact OmniAnomaly released machine-3-11 '
                'raw numeric rows. SHCL paper-era Service NPY preprocessing/version not supplied; not all SMD entities.'))
    reference='configs/runtime/fm_timegan_energy_stage_restore_audit.v1.json'
    resource=native.read(native.ROOT/reference); source_pins=[model,operator,reference,
        'configs/acquisition/fm_shcl_shared_data_retry.v1.json', acquisition,
        'scripts/flow_matching/prepare_shcl_released_data_v1.py',
        'scripts/flow_matching/run_shcl_released_data_preparation_v1.py',
        'scripts/flow_matching/register_shcl_released_data_preparation_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/heavy_scheduler_v2.py', 'scripts/flow_matching/heavy_resources.py']
    source_pins.extend('configs/datasets/mtsad_'+dataset+'.json' for dataset in ('msl','smap','psm','swat'))
    config=dict(model_config=model,reference_resource_runtime=reference,
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        environment=data_environment(),python=data_environment()['executable'],datasets=datasets,
        profiles=native.read(native.ROOT/operator)['profiles'],settings=resource['settings'],resource_lock=resource['resource_lock'],
        state_root='experiments/runs/fm_shcl_released_data_preparation_v1',
        view_root='F:/aicoding/IIA_Data/experiments/shcl_released_native_v1/data_views_v1',
        source_receipts=[dict(path=p,sha256=native.sha(native.ROOT/p)) for p in source_pins],
        boundary='All raw payloads preserved. Explicit feature/file-format adaptation only; every row retained. '
            'Actual original scaler includes validation in fit and native test truncates tails; strict full-point '
            'grouped TAB still requires a separate protocol. Original environment and missing backbones remain pending.')
    exclusive(native.ROOT/'configs/datasets/fm_shcl_released_data_views.v1.json',config)
    print(json.dumps(dict(datasets=len(datasets),full_native_loader_modes=52,raw_files_preserved=True)))


if __name__=='__main__':main()
