"""Freeze all117 source-control classification transforms without model execution."""
import json
from scripts.flow_matching import prepare_moment_icm_classification_v1 as s

CONFIG='configs/reproducibility/fm_moment_ICM_full117_classification_preprocessing.2026-10-15_v1.json'


def registration():
    scope=s.read(s.ROOT/s.PAPER_SCOPE)
    raw_audit=scope['output_directory']+'/complete_raw_data_audit.json'
    audit=s.read(s.ROOT/raw_audit)
    records={(r['dataset'],r['split']):r for r in audit['files'] if 'table' in r}
    names=list(dict.fromkeys(r['dataset'] for r in scope['all630_reported_targets'] if r['table'] in [6,7]))
    source='experiments/runs/mtsad_protocol_audit/sources/moment_research/original'
    pins=[s.PAPER_SCOPE,raw_audit,'scripts/flow_matching/prepare_moment_icm_classification_v1.py',
        'scripts/flow_matching/register_moment_icm_classification_preprocessing_v1.py',
        'scripts/flow_matching/resume_anomaly_transformer_full500_metadata_v1.py',
        'scripts/flow_matching/run_moment_full_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'tests/test_fm_moment_icm_classification_preprocessing_v1.py']
    pins += [source+'/'+p for p in ['moment/data/load_data.py','moment/data/classification_datasets.py',
        'moment/utils/data.py','scripts/zero_shot/unsupervised_representation_learning_multivariate.py']]
    return dict(schema_version=1,paper_scope=s.PAPER_SCOPE,paper_scope_sha256=s.sha(s.ROOT/s.PAPER_SCOPE),
        paper_citation='https://arxiv.org/abs/2409.13530v1',related_MOMENT_source_commit='3ab637e413f35f2c317573c0ace280d825c558de',
        related_source_is_not_original_ICM_pipeline_certification=True,
        source_root=source,raw_audit=raw_audit,sequence_length=256,environment=s.ENVIRONMENT,
        python=str(s.ROOT/'.venv/moment-author-py310/Scripts/python.exe'),
        source_normalization='original_separate_full_TRAIN_and_TEST_fits',
        train_val_merge='entire_official_TRAIN_after_source_train_val_concatenation',
        jobs=[dict(dataset=name,splits={split:dict(path=records[name,split]['logical_path'],
            sha256=records[name,split]['sha256'],instances=records[name,split]['observed_instances']) for split in ['TRAIN','TEST']}) for name in names],
        settings=dict(minimum_available_memory_bytes=16*2**30,minimum_commit_headroom_bytes=21*2**30,
            emergency_commit_headroom_bytes=16*2**30,maximum_worker_rss_bytes=3*2**30,maximum_worker_private_bytes=3*2**30),
        resource_policy='Independent non-neural data materialization:21GiBadmission =16GiBemergency+3GiBworkerprivatecap+2GiBreserve. '
            'Existing NODE/BRITS16GiB and CrossAD8GiBemergency guards and all model/replay admission budgets remain unchanged. '
            'No per-data values/channels/cases/epochs/grids/precision are reduced to meet this data-worker cap;failed or partial attempts remain failures without automatic retry.',
        resource_lock='experiments/runs/fm_bounded_strict_baseline_cpu.lock',
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_classification_source_preprocessing_v1',
        state_root='experiments/runs/fm_moment_icm_full117_classification_preprocessing_v1',
        source_receipts=[dict(path=p,sha256=s.sha(s.ROOT/p)) for p in sorted(set(pins))],
        boundary='All117original paper classification datasets and234official TRAIN/TEST files. '
            'Private AST copies of pinned MOMENT parser,classification class,Scipy resampling and normalization exclusions. '
            'Every official TRAIN sample is retained:equivalence to concatenating original train/val preprocessing tested, '
            'including a split through the middle of one multichannel instance. '
            'Native transformed values keep their source dtype;all channels and256coordinates,labels,instance order and padding masks retained. '
            'Original separate TEST standardizer fit is explicitly retained as a transductive source control,not strict train-only preprocessing. '
            'Historical ICM pipeline equivalence is unverified;independent transform rewrite not performed. '
            'No neural model training/inference,feature embedding or SVM;all original pretraining/forecast/classification results remain uncompleted.')


def main():
    path=s.ROOT/CONFIG
    if path.exists():raise FileExistsError('Preserve frozen full117 preprocessing registration')
    config=registration();s.verify(config)
    with path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(config,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(configuration=CONFIG,datasets=117,official_splits=234,new_model_runs=0)))


if __name__=='__main__':main()
