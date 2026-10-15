"""Register both Table2 direct-Max training ablations at every dataset/seed."""
from copy import deepcopy
import json

from scripts.flow_matching import run_anomaly_transformer_paper_max_ablation_v1 as runner

ROOT,read,sha=runner.ROOT,runner.read,runner.sha
CONFIG='configs/experiments/fm_anomaly_transformer_paper_max_ablation.v1.json'


def main():
    target=ROOT/CONFIG
    if target.exists():raise FileExistsError('Preserve frozen full ablation registrations')
    original=read(ROOT/runner.PARENT);runner.parent.verify(original)
    config=deepcopy(original)
    config.update(parent_registration=runner.PARENT,parent_registration_sha256=sha(ROOT/runner.PARENT),
        base_training=original['training'],variants=runner.VARIANTS,
        training=dict(original['training'],backward_strategy='one_joint_direct_Max_backward_then_one_Adam_step',
            loss_reduction='per_window_Frobenius_squared_minus_lambda_L1_joint_discrepancy_then_batch_mean'),
        primary_score_stream='all.multiplication',
        optimizer_update='one_joint_direct_Max_backward_then_one_Adam_step',
        fixed_prior_scale=1,paired_shared_parameter_initialization=True,
        artifact_root='F:/aicoding/IIA_Data/experiments/anomaly_transformer_paper_max_ablation_v1/models',
        state_root='experiments/runs/fm_anomaly_transformer_paper_max_ablation_v1',
        paper_ablation_evidence=dict(pdf_page=8,table=2,visually_verified=True,
            rows=['Assoc / Fix / Max','Assoc / Learnable / Max'],
            caption_definition='Fix sets the Gaussian scale sigma to1.0; Max is Equation4'),
        boundary='Full Table2 direct-Max reconstruction with learnable Gaussian prior and fixedsigma1prior. '
            'Exactly50new training jobs:two variants xfive real datasets xfive registered seeds. '
            'Joint Equation4 gradient through P and S,one backward/Adam step per minibatch; fixed variant '
            'removes sigma-projection trainables after consuming paired initialization RNG. '
            'All512width/three layers/batch32/window100/max10epochs,independent validation,full saved '
            'score components and unchanged memory guards retained. Parent reconstruction conventions '
            'remain explicit,with actual Max update and fixed prior overriding minimax-only choices. '
            'No exact paper/strictTAB certificate; other Table2vanillaTransformer and all remaining '
            '45/681/273minimum project experiments,replays and uncertainty remain required.')
    config['explicit_reconstruction_choices']=[choice for choice in original['explicit_reconstruction_choices']
        if 'Equation5' not in choice and 'two-loss checkpoint criterion' not in choice and 'Gaussian positivity' not in choice]
    config['explicit_reconstruction_choices'] += [
        'Apply Equation4 directly: per-window Frobenius reconstruction squared minus3 times joint symmetric-KL L1, then batch mean',
        'Both P and S remain differentiable in direct Max; exactly one autograd execution and one Adam step per minibatch',
        'Retain source FFN512 and sigma positivity mapping only for the learnable-prior branch',
        'Fixed branch bypasses source sigma transforms to exactGaussian sigma1; no sigma-projection parameters remain in optimizer/checkpoint',
        'Consume the discarded sigma-projection initialization RNG so every shared initial weight matches the learnable branch for the same seed',
        'Retain parent patience3 and LR schedule; both validation-loss slots carry the same direct-Max loss, reducing checkpoint selection to one criterion',
        'Each variant is fully retrained; reusing minimax checkpoints or changing score postprocessing cannot substitute for these ablations',
        'Prespecified primary score is multiplication; no test-label selection of the best alternative criterion']
    config['jobs']=[dict(id=f"anomaly_transformer__paper_ablation__{v}__{j['dataset'].lower()}__seed{j['seed']}",
                        dataset=j['dataset'],seed=j['seed'],variant=v)
                    for v in runner.VARIANTS for j in original['jobs']]
    files=[runner.PARENT,'scripts/flow_matching/run_anomaly_transformer_paper_max_ablation_v1.py',
        'scripts/flow_matching/register_anomaly_transformer_paper_max_ablation_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/run_light_controller.py',
        'tests/test_fm_anomaly_transformer_paper_max_ablation_v1.py',
        'configs/models/fm_anomaly_transformer_paper_max_ablation.v1.json']
    config['source_receipts'] += [dict(path=p,sha256=sha(ROOT/p)) for p in files]
    runner.verify(config)
    with target.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(config,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(registered_full_training_jobs=50,variants=runner.VARIANTS,
                          actual_autograd_executions_per_optimizer_update=1,configuration=CONFIG)))


if __name__=='__main__':main()
