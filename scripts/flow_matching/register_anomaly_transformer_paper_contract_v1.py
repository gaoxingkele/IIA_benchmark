"""Freeze all five real datasets/five seeds for the primary-paper reconstruction."""
import json
from pathlib import Path
import sys

from scripts.flow_matching import run_anomaly_transformer_paper_contract_v1 as runner

ROOT,read,sha=runner.ROOT,runner.read,runner.sha
CONFIG='configs/experiments/fm_anomaly_transformer_paper_contract.v1.json'
SOURCE='experiments/runs/fm_baseline_code_completion/sources/anomaly_transformer_author/original'


def main():
    target=ROOT/CONFIG
    if target.exists():raise FileExistsError('Preserve frozen registrations')
    contract=read(ROOT/runner.CONTRACT)
    config=dict(schema_version=1,contract=runner.CONTRACT,contract_sha256=sha(ROOT/runner.CONTRACT),
        source_directory=SOURCE,source_repository='https://github.com/thuml/Anomaly-Transformer',
        source_commit='b0ee470c8012bff857fb600462aec6209c4a18d9',
        model=dict(d_model=512,d_ff=512,n_heads=8,e_layers=3,dropout=0.0,window=100),
        training=dict(batch_size=32,maximum_epochs=10,window_stride=100,learning_rate=.0001,
            discrepancy_weight=3,patience=3,kl_log_epsilon=.0001,cpu_threads=2,optimizer='Adam',
            backward_strategy='two_detached_losses_one_accumulated_Adam_update',
            loss_reduction='per_window_Frobenius_squared_and_L1_discrepancy_then_batch_mean',
            learning_rate_schedule='inherited_author_halving_after_epoch_two'),
        python=str(Path(sys.executable)),seeds=list(range(1103,1108)),score_streams=runner.STREAMS,
        original_scope=contract['original_scope'],data_root='data/public_datasets/mtsad',
        data_cache_root='F:/aicoding/IIA_Data/experiments/anomaly_transformer_paper_contract_v1/derived_data',
        artifact_root='F:/aicoding/IIA_Data/experiments/anomaly_transformer_paper_contract_v1/models',
        state_root='experiments/runs/fm_anomaly_transformer_paper_contract_v1',
        resource_lock='experiments/runs/fm_heavy_recovery_cpu.lock',
        settings=dict(minimum_available_memory_bytes=32*1024**3,minimum_commit_headroom_bytes=40*1024**3,
                      emergency_commit_headroom_bytes=16*1024**3),
        jobs=[dict(id=f'anomaly_transformer__paper_contract_reconstruction__{d.lower()}__seed{s}',dataset=d,seed=s)
              for d in ['SMD','MSL','SMAP','PSM','SWaT'] for s in range(1103,1108)],
        explicit_reconstruction_choices=[
            'Use contiguous first80percent training and independent validation tail: all five Table13 counts match; historical ordering not certified',
            'StandardScaler fit on training subset only; inherit PSM NaN-to-zero source preprocessing',
            'Inherit512-wide FFN and source Gaussian positivity transformation; paper does not specify these executable details',
            'Use source1e-4 log epsilon for numerical stability; mathematical KL has no specified epsilon',
            'Use Equation5 per-window Frobenius squared and L1 sums with batch mean, accumulate both detached phase gradients in one Adam step',
            'Inherit patience3/two-loss checkpoint criterion and author learning-rate halving; paper specifies early stopping within10 but not these details',
            'Use Algorithm3 feature mean in the scoring criterion; Equation6 feature-sum presentation is separately noted',
            'Use Algorithm2 head means before KL, mean layers and no unreported temperature50',
            'Seven validation-calibrated ROC operating points are saved; their curve integration is not certified as Figure3 AUC',
            'CPU reproduction; original reported GPU efficiency is not reproduced by these timings'],
        remaining_original_requirements=contract['required_experiment_groups'],
        boundary='Explicit primary-paper reconstruction, separate from frozen author-runtime and restored-loader '
            'tracks. Full five-real-dataset/five-seed main matrix,512width,batch32,non-overlap100,max10epochs; '
            'independent training/validation split and validation-only checkpoint/threshold selection. '
            'Full seven criterion/layer streams on train/validation/test are retained. Original NeurIPS-TS '
            'data/experiment, all training ablations and sensitivity runs, adjacent-weight studies, baselines, '
            'independent replay and uncertainty remain. Explicit unresolved executable conventions prevent '
            'paper-equivalence certification; no strict TAB or complete45/681/273claim.')
    tracked=[ROOT/runner.CONTRACT,ROOT/contract['paper'],Path(__file__),
             ROOT/'scripts/flow_matching/run_anomaly_transformer_paper_contract_v1.py',
             ROOT/'scripts/flow_matching/run_anomaly_transformer_author_full_v1.py',
             ROOT/'scripts/flow_matching/heavy_resources.py',ROOT/'scripts/flow_matching/heavy_scheduler_v2.py',
             ROOT/'src/iia_benchmark/evaluation/mtsad_metrics.py',
             ROOT/'tests/test_fm_anomaly_transformer_paper_contract_v1.py',
             ROOT/'configs/models/fm_anomaly_transformer_paper_contract.v1.json']
    tracked+=sorted((ROOT/SOURCE/'model').glob('*.py'))
    for dataset in ['SMD','MSL','SMAP','PSM','SWaT']:
        path=ROOT/f'configs/datasets/mtsad_{dataset.lower()}.json';tracked.append(path);ds=read(path)
        tracked += [ROOT/config['data_root']/ds['directory']/name for name in ds['files'].values()]
    config['source_receipts']=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in sorted(set(tracked))]
    runner.verify(config)
    with target.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(config,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(registered_main_jobs=25,complete_score_streams_per_split=7,
        maximum_epochs=10,batch_size=32,paper_equivalence_certified=False,config=CONFIG)))


if __name__=='__main__':main()
