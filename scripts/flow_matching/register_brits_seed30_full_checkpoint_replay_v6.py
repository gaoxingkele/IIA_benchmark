"""Finish the original five-seed cohort through unchanged full CPU model replay."""
from datetime import datetime,timezone
from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT,read,sha,verify_receipts
from scripts.flow_matching.run_flow_mismatching_full_image_evaluation_v1 import write_new
from scripts.flow_matching.audit_saits_full_checkpoint_replay_v1 import verify_registration


def main():
    previous='configs/reproducibility/fm_brits_electricity_seed29_full_checkpoint_replay.2026-10-11_v5.json'
    config=read(ROOT/previous);verify_receipts(config['source_receipts'])
    identifier='saits_paper_brits_electricity_author_missing0.1_seed30'
    native='configs/experiments/fm_saits_timeseries_jobs/'+identifier+'.json'
    output='experiments/runs/flow_matching_campaign/'+identifier
    job=read(ROOT/native);result=read(ROOT/output/'result.json')
    if (result['status']!='completed' or result['config']!=job or job['seed']!=30
            or result['config_sha256']!=sha(ROOT/native) or result['parameters']!=6993328
            or result['target_count']!=1020954 or result['test_windows']!=290
            or job['diagnostic'] or result['dataset_sha256']!=job['dataset_sha256']):
        raise ValueError('Complete original full seed30 result/data/job identity required')
    config.update(completed_jobs=[dict(id=identifier,config=native,result_path=output+'/result.json',result_sha256=sha(ROOT/output/'result.json'))],
        artifact_directory='F:/aicoding/IIA_Data/experiments/brits_electricity_seed30_full_checkpoint_replay_v6',
        state_root='experiments/runs/fm_brits_electricity_seed30_full_checkpoint_replay_v6',
        original_replay_registration=dict(path=previous,sha256=sha(ROOT/previous)),
        boundary=config['boundary'].replace('seed29','seed30'))
    old_paths={x['path'] for x in config['source_receipts']}
    for path in [previous,native,*[output+'/'+name for name in ('result.json','resume.pt','best.pt','group_statistics.npz')]]:
        if path not in old_paths:config['source_receipts'].append(dict(path=path,sha256=sha(ROOT/path)));old_paths.add(path)
    verify_registration(config)
    for field in ('artifact_directory','state_root'):
        if (ROOT/config[field]).exists():raise FileExistsError('Preserve any earlier original seed30 full replay attempt')
    path='configs/reproducibility/fm_brits_electricity_seed30_full_checkpoint_replay.2026-10-11_v6.json'
    write_new(ROOT/path,config)
    target='projects/flow_matching_research/results/2026-10-11/brits_electricity_seed30_full_replay_v6'
    write_new(ROOT/target/'registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=path,configuration_sha256=sha(ROOT/path),original_result_path=output+'/result.json',
        original_result_sha256=sha(ROOT/output/'result.json'),actual_result_status=result['status'],
        model_parameters=6993328,native_test_windows=290,held_out_coordinates=1020954,
        original_metrics_pending_independent_full_replay=result['metrics'],
        original_validation_selected_epoch=result['validation_best']['epoch'],
        original_scope=config['original_scope'],original_memory_gates_and_metric_tolerance_preserved=True,
        author_equivalence_certified=False,all_original_experiments_complete=False,
        original_training_seeds_completed=[26,27,28,29,30],original_training_seeds_full_replay_verified=[26,27,28,29],
        required_original_training_seeds=5,new_published_benchmark_result_rows=0))
    print(dict(registered_full_seed30_replay=True,all_original_training_seeds_completed=5,independent_verified_seeds=4))


if __name__=='__main__':main()
