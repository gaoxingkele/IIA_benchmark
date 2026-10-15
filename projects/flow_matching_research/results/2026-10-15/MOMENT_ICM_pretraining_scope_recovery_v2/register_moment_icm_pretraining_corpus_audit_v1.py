"""Freeze the full public snapshot and every related-source getter input."""
import json
from pathlib import Path
from scripts.flow_matching import audit_moment_icm_pretraining_corpus_v1 as s

CONFIG='configs/reproducibility/fm_moment_ICM_pretraining_corpus_audit.2026-10-15_v1.json'


def registration():
    scope_path='configs/reproducibility/fm_moment_ICM_primary_scope_data.2026-10-15_v1.json'
    prepared_path='configs/reproducibility/fm_moment_ICM_full117_classification_preprocessing.2026-10-15_v1.json'
    scope=s.read(s.ROOT/scope_path);prepared=s.read(s.ROOT/prepared_path)
    source=prepared['source_root'];pile=Path(scope['classification_root']).parents[1].as_posix()
    groups,compat=s.inputs.enumerate_inputs(s.ROOT/source,s.ROOT/pile)
    pins=[scope_path,prepared_path,scope['pile_integrity'],
        'scripts/flow_matching/moment_icm_pretraining_inputs_v1.py',
        'scripts/flow_matching/audit_moment_icm_pretraining_corpus_v1.py',
        'scripts/flow_matching/register_moment_icm_pretraining_corpus_audit_v1.py',
        'scripts/flow_matching/prepare_moment_icm_classification_v1.py',
        'scripts/flow_matching/resume_anomaly_transformer_full500_metadata_v1.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'tests/test_fm_moment_icm_pretraining_inputs_v1.py']
    pins += [source+'/'+p for p in ['moment/data/dataloader.py','moment/data/classification_datasets.py',
        'moment/data/forecasting_datasets.py','moment/data/anomaly_detection_datasets.py','moment/data/load_data.py',
        'moment/utils/data.py','configs/default.yaml','configs/pretraining/pretrain.yaml',
        'scripts/zero_shot/unsupervised_representation_learning_multivariate.py']]
    return dict(schema_version=1,paper_citation='https://arxiv.org/abs/2409.13530v1',
        related_source_commit='3ab637e413f35f2c317573c0ace280d825c558de',source_root=source,pile_root=pile,
        integrity=scope['pile_integrity'],snapshot_files=2441,monash_inputs=47,context_length=256,source_seed=13,
        original_paper_pretraining_epochs=1,data_source_fraction=1.,environment=prepared['environment'],python=prepared['python'],
        all_source_getter_paths={k:[Path(p).relative_to(s.ROOT).as_posix() for p in v] for k,v in groups.items()},
        path_compatibility_observation=compat,
        settings=dict(minimum_available_memory_bytes=16*2**30,minimum_commit_headroom_bytes=24*2**30,
            emergency_commit_headroom_bytes=16*2**30,maximum_worker_rss_bytes=6*2**30,maximum_worker_private_bytes=6*2**30),
        resource_policy='New non-neural entire-snapshot/cardinality audit:24GiBadmission=16GiBemergency+6GiBprivatecap+2GiBreserve. '
            'All original model and replay guards/budgets/tolerances unchanged. One source Monash frame at a time,all47files retained.',
        resource_lock=prepared['resource_lock'],
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_pretraining_input_scope_v1',
        state_root='experiments/runs/fm_moment_icm_pretraining_input_scope_v1',
        source_receipts=[dict(path=p,sha256=s.sha(s.ROOT/p)) for p in sorted(set(pins))],
        boundary='Rehash all2441registered public PILE files,enumerate all related-source classification/autoformer/Monash/optionalFRED/anomaly paths, '
            'profile every47Monash file using actual pinned source parser/class in exact producer environment with scaleFalse for cardinalities only. '
            'Original source length field is dataframe row count;compare its gate with a separate observed-training-time-length gate. '
            'Do not silently correct the historical source track,pool controls,subsample datasets,or call different input/token budgets an isolated model ablation. '
            'Forward-slash getter facade preserves explicit CharacterTrajectories exclusion;nativeWindows mismatch retained. '
            'Absent optionalFREDbranch is recorded,not automatically claimed paper-required missing data;FRED-MD remains a separate Monash input. '
            'Profiles do not execute dataset random window sampling,normalization,neural training,inference or SVM;1epoch remains unexecuted. '
            'Current public snapshot integrity is not historical ICM corpus/version equivalence or a full45/681/273completion certificate.')


def main():
    target=s.ROOT/CONFIG
    if target.exists():raise FileExistsError('Preserve frozen entire-corpus audit registration')
    config=registration();s.verify(config)
    with target.open('x',encoding='utf-8',newline='\n') as stream:json.dump(config,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(configuration=CONFIG,raw_files=2441,Monash_inputs=47,
        source_family_counts={k:len(v) for k,v in config['all_source_getter_paths'].items()},pretraining_executed=0)))


if __name__=='__main__':main()
