"""Register the full Table2 plain Transformer reconstruction baseline."""
from copy import deepcopy
import json

from scripts.flow_matching import run_plain_transformer_paper_baseline_v1 as runner

ROOT,read,sha=runner.ROOT,runner.read,runner.sha
CONFIG='configs/experiments/fm_plain_transformer_paper_baseline.v1.json'


def main():
    target=ROOT/CONFIG
    if target.exists():raise FileExistsError('Preserve frozen complete baseline registration')
    original=read(ROOT/runner.PARENT);runner.parent.verify(original)
    config=deepcopy(original)
    config.update(parent_registration=runner.PARENT,parent_registration_sha256=sha(ROOT/runner.PARENT),
        base_training=original['training'],training=dict({k:v for k,v in original['training'].items()
            if k not in ['discrepancy_weight','kl_log_epsilon']},
            backward_strategy='one_reconstruction_backward_then_one_Adam_step',
            loss_reduction='per_window_Frobenius_squared_then_batch_mean'),
        score_streams=runner.STREAMS,primary_score_stream='all.reconstruction',prior_path_present=False,
        paired_shared_parameter_initialization=True,
        artifact_root='F:/aicoding/IIA_Data/experiments/plain_transformer_paper_baseline_v1/models',
        state_root='experiments/runs/fm_plain_transformer_paper_baseline_v1',
        paper_baseline_evidence=dict(pdf_page=8,table=2,row='Transformer / Recon / no prior / no discrepancy optimization'),
        boundary='Full Table2 plain Transformer reconstruction baseline:25new training jobs,five real datasets '
            'xfive registered seeds. No Gaussian prior computation,distance matrices or sigma-projection '
            'trainables; shared scaled-dot-product attention/embedding/FFN initialization paired to the parent. '
            'Only reconstruction loss and reconstruction score; entropy arrays are diagnostic only. '
            'All512width/three layers/eight heads/batch32/window100/max10epochs,independent validation, '
            'training-only normalization,complete covered timestamps and resource guards retained. '
            'Inherited explicit architecture/normalization/optimization conventions and baseline-specific '
            'historical tuning remain uncertified. Not the lambda0association-scored Anomaly Transformer. '
            'Other paper experiments,replays,uncertainty,strictTAB and full45/681/273minimumscope remain.')
    config['jobs']=[dict(id=f"plain_transformer__paper_Table2__{j['dataset'].lower()}__seed{j['seed']}",
        dataset=j['dataset'],seed=j['seed'],variant='plain_attention_reconstruction') for j in original['jobs']]
    config['explicit_reconstruction_choices']=[choice for choice in original['explicit_reconstruction_choices']
        if not any(s in choice for s in ['Equation5','two-loss checkpoint criterion','Gaussian positivity','log epsilon','Algorithm2'])]
    config['explicit_reconstruction_choices'] += [
        'Retain the parent full512wide Transformer embedding/positional encoding,FFN512 and three-layer encoder conventions',
        'Remove all prior computations and trainables; consume discarded sigma-projection initialization RNG to pair shared weights',
        'Use one Frobenius-squared reconstruction backward and one Adam step per minibatch',
        'Retain inherited patience3 and LR schedule; both validation slots contain identical reconstruction loss',
        'Score only Algorithm3mean feature reconstruction error; no association criterion is evaluated or saved',
        'Save mean-head attention entropy as an explicitly named diagnostic,not association discrepancy',
        'Use the same predeclared validation ratio as the comparison contract; historical baseline-specific tuning is unavailable',
        'The parent fixed-delta thresholds are retained only as explicitly labeled diagnostics for this reconstruction baseline']
    files=[runner.PARENT,'scripts/flow_matching/run_plain_transformer_paper_baseline_v1.py',
        'scripts/flow_matching/register_plain_transformer_paper_baseline_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_paper_max_ablation_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/run_light_controller.py',
        'tests/test_fm_plain_transformer_paper_baseline_v1.py',
        'configs/models/fm_plain_transformer_paper_baseline.v1.json']
    config['source_receipts'] += [dict(path=p,sha256=sha(ROOT/p)) for p in files]
    runner.verify(config)
    with target.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(config,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(registered_full_training_jobs=25,prior_path_present=False,
                          primary_criterion='reconstruction',configuration=CONFIG)))


if __name__=='__main__':main()
