"""Freeze20 full LS4 baseline runs; retain other LS4/SaShiMi obligations."""
import argparse
import json
from pathlib import Path

import yaml

from scripts.flow_matching.run_ls4_original_v1 import ROOT, read, sha, budget, derive_config, verify
from scripts.flow_matching.spectral_table2_bootstrap_v1 import write


def register(settings):
    queue_path=ROOT/settings['queue_path']
    if queue_path.exists():
        raise FileExistsError('Keep frozen LS4 registrations immutable')
    lock=read(ROOT/settings['environment_lock'])
    if not lock['passed'] or lock['python_version'].split('.')[:2]!=['3','10']:
        raise ValueError('Exact original LS4 environment not verified')
    datasets={}
    for data in settings['datasets']:
        data=dict(data,raw_sha256=sha(ROOT/data['raw_path']))
        datasets[data['dataset']]=data
    jobs=[]
    for seed in settings['seeds']:
        for protocol in ('released','paper_literal'):
            for dataset,data in datasets.items():
                source_config=str(Path(settings['source_root'])/'configs/monash'/data['yaml']).replace('\\','/')
                template=yaml.safe_load((ROOT/source_config).read_text(encoding='utf-8'))
                settings_model=dict(protocol=protocol)
                derived=derive_config(template,settings_model,'PRIVATE_AUTHOR_DATA_PATH')
                ident=f'ls4_original__{protocol}__{dataset}__seed{seed}'
                path='configs/models/fm_'+ident+'.json'
                model=dict(id=ident,algorithm='LS4',dataset=dataset,protocol=protocol,seed=seed,
                    task='unconditional_time_series_generation',source_config=source_config,
                    samples=data['samples'],length=data['length'],train_samples=int(data['samples']*.8),
                    test_samples=data['samples']-int(data['samples']*.8),optim=derived['optim'],
                    sigma=derived['model']['sigma'],diagnostic=False,
                    implementation='scripts.flow_matching.run_ls4_original_v1:run_job',
                    reproduction_status='full_native_budget_registered_not_completed',citations=settings['citations'],
                    seed_boundary='Five local paired robustness seeds; historical author seeds unspecified',
                    boundary='Exact released source models/optimizer/EMA/full trajectory data/evaluators. Paper-literal overrides only epochs7000,batch64,sigma0.1. Each final metric uses last EMA/model; source best-test-marginal checkpoint is saved separately. Full-series normalization and80/20 trajectory split are original source behavior, not strict TSAD.',
                    metrics=dict(clf_score='Original100-epoch S4D classifier BCE loss, not accuracy or F1',
                        marginal_score='Original50-bin histogram marginal loss',
                        predictive_score='Original100-epoch S4D10-step predictor MSE'))
                model['budget']=budget(model)
                if (ROOT/path).exists():
                    raise FileExistsError('Preserve previous model registration')
                write(ROOT/path,model)
                jobs.append(dict(id=ident,model_config=path,model_config_sha256=sha(ROOT/path),
                    artifact_directory=str(Path(settings['artifact_root'])/ident),
                    output_directory=settings['state_root']+'/jobs/'+ident))
    queue={key:value for key,value in settings.items() if key not in ('datasets','artifact_root')}
    queue.update(datasets=datasets,jobs=jobs,source_receipts=[])
    paths={settings['environment_lock'],settings['registration_path'],
        'scripts/flow_matching/run_ls4_original_v1.py','scripts/flow_matching/run_ls4_original_queue_v1.py',
        'scripts/flow_matching/register_ls4_original_v1.py','tests/test_fm_ls4_original_v1.py',
        'scripts/flow_matching/full_gpu_memory_slot_v2.py','scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py'}
    paths.update(str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/settings['source_root']).rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    paths.update(data['raw_path'] for data in datasets.values())
    paths.update(j['model_config'] for j in jobs)
    queue['source_receipts']=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(paths)]
    verify(queue);write(queue_path,queue)
    print(json.dumps(dict(full_jobs=len(jobs),generator_updates=sum(read(ROOT/j['model_config'])['budget']['generator_updates'] for j in jobs),
        queue_sha256=sha(queue_path),source_bindings=len(queue['source_receipts']))))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();settings=read(ROOT/args.config);register(settings)


if __name__=='__main__':
    main()
