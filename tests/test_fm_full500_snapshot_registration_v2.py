from pathlib import Path
from scripts.flow_matching import audit_full500_image_checkpoint_CPU_v2 as audit

CONFIG='configs/runtime/fm_full500_image_checkpoint_CPU_audit.v2.json'
ARCHIVE='configs/reproducibility/fm_full500_checkpoint_immutable_archive.2026-10-12_v2.json'


def test_exact_native_original_job_full25_and_unchanged_checkpoint_resource_guards():
    config=audit.read(audit.ROOT/CONFIG);job=audit.verify(config)
    prior=audit.read(audit.ROOT/'configs/runtime/fm_full401_image_checkpoint_CPU_audit.v1.json')
    for key in ('python','settings','resource_lock','job_configuration','original_full25_configuration','original_scope'):
        assert config[key]==prior[key]
    assert config['completed_epochs']==500 and config['required_original_epochs']==2000
    assert Path(config['checkpoint_receipt']['path']).name=='epoch_0500.pt'
    assert job['seed']==1103 and job['variant']=='conditional' and job['dataset']=='mvtec'


def test_actual_complete500_source_snapshot_and_separate_archive_bytes_match():
    config=audit.read(audit.ROOT/CONFIG);archive=audit.read(audit.ROOT/ARCHIVE)
    source=archive['source_checkpoint'];saved=archive['immutable_archive']
    assert config['checkpoint_receipt']['path']==source['path'] and source['path']!=saved['path']
    assert audit.sha(source['path'])==source['sha256']==saved['sha256']==audit.sha(saved['path'])
    assert source['bytes']==saved['bytes']==Path(saved['path']).stat().st_size
    assert archive['required_original_epochs']==2000 and archive['expected_completed_original_epochs']==500
    assert archive['original_checkpoint_untouched'] and archive['GPU_training_not_stopped']
    assert archive['new_optimizer_updates']==0 and not archive['all_original_experiments_complete']
