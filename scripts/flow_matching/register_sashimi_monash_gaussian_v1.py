"""Register all four original data tasks for explicit Gaussian AR reconstructions."""
import argparse
import copy
from pathlib import Path

from scripts.flow_matching.run_sashimi_monash_gaussian_v1 import ROOT,read,sha,verify,budget,backbone_parameters
from scripts.flow_matching.giflow_native_protocol import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();settings=read(ROOT/args.config)
    if (ROOT/settings['queue_path']).exists():raise FileExistsError('Preserve frozen Gaussian AR registration')
    datasets={};sources={}
    for queue_path in settings['original_ls4_queues']:
        q=read(ROOT/queue_path)
        data=q['datasets'].values() if isinstance(q['datasets'],dict) else q['datasets']
        for d in data:
            value=copy.deepcopy(d);value['raw_sha256']=sha(ROOT/value['raw_path']);datasets[value['dataset']]=value
        for item in q['source_receipts']:sources[item['path']]=item
    queue=dict(settings,datasets=datasets,jobs=[],source_receipts=[])
    queue['registration_path']=args.config
    artifact=Path(settings['artifact_root'])
    for name in ['data_audit','integration','gpu_preflight']:
        queue[name+'_artifact']=str(artifact/name)
        queue[name+'_output']=settings['state_root']+'/preflight/'+name+'.json'
    queue['data_audit_progress']=settings['state_root']+'/preflight/data_audit_progress.json'
    for seed in settings['seeds']:
        for profile in settings['profiles']:
            for dataset,data in datasets.items():
                count=int(data['samples']*.8);identifier='sashimi_gaussian__'+profile+'__'+dataset+'__seed'+str(seed)
                c=dict(id=identifier,algorithm='SaShiMi-Gaussian-AR reconstruction',dataset=dataset,profile=profile,
                    seed=seed,task='unconditional_time_series_generation',samples=data['samples'],length=data['length'],
                    train_samples=count,test_samples=data['samples']-count,source_config=settings['source_root']+'/configs/monash/'+data['yaml'],
                    backbone=backbone_parameters(profile),sigma=.1,activation='sigmoid' if dataset=='temperature_rain' else 'identity',
                    optim=dict(epochs=1000 if dataset=='temperature_rain' else 7000,batch_size=64,lr=.001,weight_decay=0.,lamb=.999,start_step=200),
                    diagnostic=False,author_equivalence_certified=False,implementation='scripts.flow_matching.run_sashimi_monash_gaussian_v1:run_job',
                    reproduction_status='paper_informed_full_budget_reconstruction_registered_not_completed',
                    citations=settings['citations'],boundary=settings['boundary'],paper_reference=settings['paper_claims'][dataset],
                    profile_boundary='Primary four flat S4 blocks from paper description' if profile=='paper_flat4' else 'Local architecture sensitivity: released LS4 decoder pool[1] yields eight S4 blocks, not a paper-reported ablation',
                    metric_units=dict(clf_score='Native SUM of final classifier test-batch BCE means; higher is less distinguishable, not F1',
                        predictive_score='Native SUM of final predictive test-batch MSE means; tail10 teacher-forced one-step targets',
                        marginal_score='Original50-bin normalized histogram density discrepancy'),
                    evaluation_boundary='Both final model and final EMA use paired saved evaluator RNG, full original100-epoch evaluators. Baseline-specific evaluation cadence/selection unpublished; final-only evaluation is an explicit reconstruction choice.')
                c['budget']=budget(c)
                path='configs/models/fm_'+identifier+'.json'
                if (ROOT/path).exists():raise FileExistsError('Preserve original Gaussian AR model config')
                write_json(ROOT/path,c);sources[path]=dict(path=path,sha256=sha(ROOT/path))
                queue['jobs'].append(dict(id=identifier,model_config=path,model_config_sha256=sha(ROOT/path),
                    artifact_directory=str(artifact/identifier),output_directory=settings['state_root']+'/jobs/'+identifier))
    paths=[args.config,settings['cauchy_correction_config'],settings['paper_pdf'],
        'experiments/runs/fm_spectral_table3_baseline_acquisition_v1/sources/sashimi_s4/original/src/models/functional/cauchy.py',
        'scripts/flow_matching/ls4_cauchy_fallback_repair_v1.py',
        'scripts/flow_matching/sashimi_gaussian_adapter_v1.py','scripts/flow_matching/run_sashimi_monash_gaussian_v1.py',
        'scripts/flow_matching/run_sashimi_monash_gaussian_queue_v1.py','scripts/flow_matching/register_sashimi_monash_gaussian_v1.py',
        'tests/test_fm_sashimi_monash_gaussian_v1.py']+settings['original_ls4_queues']
    for p in paths:sources[p]=dict(path=p,sha256=sha(ROOT/p))
    queue['source_receipts']=list(sources.values());verify(queue)
    if sum(read(ROOT/j['model_config'])['budget']['generator_updates'] for j in queue['jobs'])!=4430000:
        raise ValueError('All original full Gaussian AR generator budgets required')
    write_json(ROOT/settings['queue_path'],queue)
    print('Registered40 full paper-informed Gaussian AR slots,4430000 generator updates; no completed performance implied')


if __name__=='__main__':main()
