"""Freeze the released SHCL VAE and all413 original script commands.

Current author release and local reconstruction remain distinct. Registering a
native callable/complete script grid does not certify data, environment or paper
equivalence and does not mark any training or reported score complete.
"""
from datetime import datetime,timezone
import json

from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    source='experiments/runs/mtsad_protocol_audit/sources/shcl/original'
    snapshot='experiments/runs/mtsad_protocol_audit/sources/shcl/snapshot.json'
    metadata=read(ROOT/snapshot)
    if metadata['commit']!='022520104f2d0d3bda563557ebc99279e6812511':
        raise ValueError('Exact preserved current author release required')
    files=[p.relative_to(ROOT).as_posix() for p in (ROOT/source).rglob('*') if p.is_file()
           and p.suffix in ('.py','.sh','.md','.txt')]
    files.extend([snapshot,'scripts/flow_matching/shcl_released_native_v1.py'])
    config=dict(schema_version=1,id='fm_shcl_released_vae_native_v1',paper_id='shcl',family='author_runtime',
        task='time_series_anomaly_detection',entrypoint='scripts.flow_matching.shcl_released_native_v1:build_model',
        native_training_entrypoint='scripts.flow_matching.shcl_released_native_v1:execute_author_command',
        source_root=source,source_commit=metadata['commit'],source_snapshot=snapshot,
        citation='https://doi.org/10.1016/j.ins.2025.122680',code='https://github.com/Zhangzzbzzb/SHCL',
        data_citations=['https://github.com/NetManAIOps/OmniAnomaly','https://github.com/khundman/telemanom',
            'https://github.com/eBay/RANSynCoders'],
        reproduction_status='released_author_VAE_callable_original_environment_data_and_full_training_equivalence_pending',
        runtime_output_root='F:/aicoding/IIA_Data/experiments/shcl_released_native_v1',
        constructor_parameters=dict(win_size=110,seq_size=5,c_in=25,c_out=25,d_model=128,e_layers=3,fr=.4,tr=.4,device='cpu'),
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(files)],
        score_contract='Untouched native core returns single/pair six-tensor tuples; original compute_loss produces[B,T]. '
            'Both train and eval sample VAE latent noise. Native test source uses within-window softmax, native validation '
            'percentile and truncated nonoverlapping test prefix; strict full-point TAB adapter remains a separate obligation.',
        runtime_correction='Only one unused --threshold CLI declaration added in memory; Solver percentile threshold unchanged.',
        boundary='Current author refactored VAE release, not established paper-era Transformer/LSTM implementation. '
            'Existing local deterministic-mean VAE reconstruction stays separate. All actual release script budgets '
            'remain obligations. Original package lock/data format and full native training/evaluation/replay remain '
            'unverified. Untrained operator checks are smoke plumbing, never benchmark performance.')
    from scripts.flow_matching.shcl_released_native_v1 import verify,released_script_commands
    verify(config);commands=released_script_commands(config)
    path='configs/models/fm_shcl_released_vae_native.v1.json'
    exclusive(ROOT/path,config)
    registry_path='configs/experiments/fm_shcl_released_script_registry.v1.json'
    registry=dict(schema_version=1,model_config=path,model_config_sha256=sha(ROOT/path),original_scope=dict(
        papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        source_commit=config['source_commit'],full_original_command_count=413,full_training_commands=49,
        full_test_commands=364,commands=commands,
        SMD_source_scope='Only author-provided machine-3-11 grid,9windows x5epoch settings x8test ratios; not all SMD machines.',
        other_original_datasets=['MSL','SMAP','PSM','SWaT'],
        source_seed=1,train_and_test_are_separate_seeded_processes=True,
        checkpoints_must_be_isolated_per_training_configuration=True,
        original_source_shared_cpt_overwrite_during_grid_not_safe_for_archival=True,
        data_views_prepared=False,environment_aligned=False,full_original_commands_executed=False,
        all_reported_paper_datasets_covered=False,paper_performance_reproduced=False,
        boundary='Exact provided script command enumeration. Data paths/physical device/filesystem output adaptation '
            'and compatible original numerical environment require audited bindings before dispatch. No command is '
            'dropped because its train/test budget is expensive. The unused threshold flags are retained as metadata '
            'without changing Solver computation. Missing other backbones/full-dataset paper protocols remain gaps.')
    exclusive(ROOT/registry_path,registry)
    base=ROOT/'projects/flow_matching_research/results/2026-10-12/shcl_released_native_preparation_v1'
    exclusive(base/'registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        model_config=path,model_config_sha256=sha(ROOT/path),script_registry=registry_path,
        script_registry_sha256=sha(ROOT/registry_path),original_scope=registry['original_scope'],
        released_source_methods_callable=True,original_train_commands=49,original_test_commands=364,
        original_script_seed_and_windows_epoch_grid_preserved=True,
        original_source_files_modified=False,original_training_executed=False,new_benchmark_results=0,
        all_reported_SHCL_backbones_complete=False,all_original_experiments_complete=False))
    print(json.dumps(dict(model_config=path,full_original_commands=413,train_commands=49,test_commands=364)))


if __name__=='__main__':main()
