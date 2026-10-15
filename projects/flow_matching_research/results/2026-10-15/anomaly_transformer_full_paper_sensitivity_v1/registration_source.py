"""Freeze all Figure7/Tables8-9 points and reuse the25 main model identities."""
from copy import deepcopy
import json
from pathlib import Path

from scripts.flow_matching import run_anomaly_transformer_paper_sensitivity_v1 as runner

ROOT,read,sha=runner.ROOT,runner.read,runner.sha
CONFIG='configs/experiments/fm_anomaly_transformer_paper_sensitivity.v1.json'


def main():
    target=ROOT/CONFIG
    if target.exists():raise FileExistsError('Keep frozen sensitivity registrations')
    original=read(ROOT/runner.PARENT);runner.parent.verify(original)
    config=deepcopy(original)
    config.pop('score_streams')
    config.update(parent_registration=runner.PARENT,parent_registration_sha256=sha(ROOT/runner.PARENT),
        axes=runner.AXES,base=runner.BASE,base_model_reuse=original['jobs'],
        score_stream_policy='all_four_criteria_plus_every_individual_layer',
        score_streams_by_layers={str(n):runner.streams(n) for n in runner.AXES['layers']},
        primary_score_stream='all.multiplication',
        artifact_root='F:/aicoding/IIA_Data/experiments/anomaly_transformer_paper_sensitivity_v1/models',
        state_root='experiments/runs/fm_anomaly_transformer_paper_sensitivity_v1',
        axis_evidence=dict(window=dict(pdf_page=13,figure=7,visually_verified_values=runner.AXES['window']),
                           weight=dict(pdf_page=13,figure=7,visually_verified_values=runner.AXES['weight']),
                           layers=dict(pdf_page=18,table=8,visually_verified_values=runner.AXES['layers']),
                           hidden=dict(pdf_page=18,table=9,visually_verified_values=runner.AXES['hidden'])),
        complete_grid_counts=dict(axis_display_slots=500,new_unique_training_jobs=400,
                                 reused_unique_main_models=25,total_unique_models=425),
        boundary='Complete primary-paper Figure7/Tables8-9 single-factor sensitivity reconstruction: '
            'window50..300by50,lambda0..6,layers1..4,hidden256/512/1024. Exactly400new full training '
            'jobs and25reused main models; no duplicate baseline training. All five real datasets and '
            'five registered seeds,full10epoch/early-stop rule,batch32,independent validation,per-window '
            'mathematical losses,all scoring components and all resource guards retained. Fixed FFN512 '
            'and other executable choices remain inherited/disclosed. No selecting the best test point '
            'as a strict leaderboard result. CPU timings do not reproduce original GPU efficiency. '
            'All other full45/681/273 experiments,NeurIPS-TS,other trained ablations,independent replay '
            'and uncertainty remain required; no exact paper or strict TAB certificate.')
    config['explicit_reconstruction_choices'] += [
        'Exactly one displayed paper axis changes per new job; primary score is fixed before any test result',
        'Window changes bind all seven original worker coverage/index/tail sites and attention window consistently',
        'Depth changes bind constructor, all individual-layer streams and component reshape together',
        'Hidden-width changes retain FFN512 per the inherited source convention; paper FFN convention remains uncertified',
        'Loss-weight0 retains the association-based score but both detached training losses reduce to reconstruction',
        'All split caches reuse identical parent training-only normalization and80/20partition, independent of window']
    config['jobs']=[]
    for axis,value in runner.variants():
        for job in original['jobs']:
            values=dict(runner.BASE);values[axis]=value
            config['jobs'].append(dict(id=f"anomaly_transformer__paper_sensitivity__{axis}{value}__{job['dataset'].lower()}__seed{job['seed']}",
                dataset=job['dataset'],seed=job['seed'],axis=axis,value=value,parameters=values))
    tracked=[runner.PARENT,'scripts/flow_matching/run_anomaly_transformer_paper_sensitivity_v1.py',
             'scripts/flow_matching/register_anomaly_transformer_paper_sensitivity_v1.py',
             'tests/test_fm_anomaly_transformer_paper_sensitivity_v1.py',
             'configs/models/fm_anomaly_transformer_paper_sensitivity.v1.json']
    config['source_receipts'] += [dict(path=p,sha256=sha(ROOT/p)) for p in tracked]
    runner.verify(config)
    with target.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(config,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(new_full_training_jobs=400,reused_main_models=25,all_unique_models=425,
                          nonbase_variants=16,axes=runner.AXES,configuration=CONFIG)))


if __name__=='__main__':main()
