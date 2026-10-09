"""Freeze all fourteen paired five-seed Table1 generation cohorts and claims."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import sys

from scripts.flow_matching.prepare_spectral_original_data import ROOT,read,sha,write


def table1_claims(cache,policy):
    page=cache['pages']['9'];table=page.split('Table 2:')[0]
    rows=[]
    metrics={'Context-FID':'context_fid','Correlational':'correlational','Discriminative':'discriminative','Predictive':'predictive'}
    names=list(metrics)
    datasets=['sines','stocks','etth','mujoco','energy','fmri']
    for i,label in enumerate(names):
        block=table.split(label+'\n',1)[1]
        if i+1<len(names):block=block.split(names[i+1]+'\n',1)[0]
        for author_name,local in [('Ours','spectral_flow'),('Diffusion-TS','diffusion_ts')]:
            lines=block.split(author_name+'\n',1)[1].splitlines()[:6]
            if len(lines)!=6:raise ValueError('Six dataset table values required')
            for dataset,line in zip(datasets,lines):
                found=re.fullmatch(r'(\d+\.\d+)±(\d*\.\d+)',line.strip())
                if not found:raise ValueError('Unrecognized primary table cell: '+line)
                rows.append({'method':local,'dataset':dataset,'metric':metrics[label],
                    'paper_mean':float(found.group(1)),'paper_reported_spread':float(found.group(2)),
                    'spread_type':'unspecified in table; author display_scores uses t-based95% half-width',
                    'source_pdf':policy['source_pdf'],'source_pdf_sha256':policy['source_pdf_sha256'],'source_page':9,'source_table':1})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/reproducibility/spectral_mean_flow_original_protocol.v2.json')
    args=parser.parse_args();policy=read(ROOT/args.config)
    path=ROOT/'configs/experiments/fm_spectral_mean_flow_original_queue.v1.json'
    if path.exists():raise FileExistsError('Preserve frozen generation queue')
    manifest_path=ROOT/policy['data_output']/'manifest.json';manifest=read(manifest_path)
    if manifest['protocol_sha256']!=sha(ROOT/args.config) or len(manifest['datasets'])!=7:
        raise ValueError('All seven paired original data cases must be frozen first')
    lock_path=ROOT/'configs/reproducibility/spectral_mean_flow_author_py310.lock.json'
    packages={distribution.metadata['Name']:distribution.version for distribution in importlib.metadata.distributions()}
    installs=[]
    for name in ['tmp/spectral_torch_install.json','tmp/spectral_dependencies_install.json','tmp/spectral_import_dependencies_install.json']:
        report=read(ROOT/name)
        installs.extend({'package':entry['metadata']['name'],'version':entry['metadata']['version'],
                         'download_info':entry['download_info']} for entry in report['install'])
    write(lock_path,{'python':sys.version,'packages':dict(sorted(packages.items())), 'wheel_receipts':installs,
        'requirements':'configs/reproducibility/spectral_mean_flow_author_py310.requirements.txt',
        'author_recommended_versions':{'torch':'2.8.0+cu126','tensorflow':'2.15.0','tensorflow-probability':'0.23.0'},
        'hardware_boundary':'Windows/RTX3090; original README tested Linux/H100','full_author_environment_equivalence':False})
    if packages['torch']!='2.8.0+cu126' or packages['tensorflow']!='2.15.0':
        raise ValueError('Required recommended dependency versions differ')
    cache_path=ROOT/'papers/literature/flow_matching/ara_text_cache/spectral_mean_flow'/f"{policy['source_pdf_sha256']}.json"
    cache=read(cache_path)
    if cache['source_sha256']!=policy['source_pdf_sha256'] or sha(ROOT/policy['source_pdf'])!=policy['source_pdf_sha256']:
        raise ValueError('Original primary PDF binding differs')
    claims=table1_claims(cache,policy)
    claims_path=ROOT/'configs/reproducibility/spectral_mean_flow_table1_claims.v1.json'
    write(claims_path,{'claims':claims,'author_script_spread_semantics':'t-based95% interval half-width, not standard deviation','local_value':None})
    import yaml
    jobs=[];model_configs=[]
    for data in manifest['datasets']:
        for record in data['artifacts']:
            if sha(ROOT/record['path'])!=record['sha256']:raise ValueError('Prepared full author data changed')
        for method in policy['models']:
            author_yaml=(ROOT/policy['author_source']/'configs'/method/f"{data['id']}.yaml")
            spec=yaml.safe_load(author_yaml.read_text(encoding='utf-8'))
            table=policy['table1'][data['id']]
            original_data=yaml.safe_load((ROOT/policy['author_source']/'configs/spectral_flow'/f"{data['id']}.yaml").read_text(encoding='utf-8'))['dataloader']['train_dataset']
            if spec['dataloader']['train_dataset']!=original_data:
                raise ValueError('Baseline original preprocessing differs; do not silently substitute a paired cache')
            if spec['model']['params']['feature_size']!=data['shape'][2] or spec['model']['params']['seq_length']!=data['shape'][1]:
                raise ValueError('Complete source data dimensions differ from the author model')
            if method=='spectral_flow':
                for key,expected in [('feature_size',table['features']),('d_model',table['hidden']),('n_layers',table['layers']),('sampling_timesteps',table['sampling_steps'])]:
                    if spec['model']['params'][key]!=expected:raise ValueError('Original YAML differs from Table6: '+key)
                if spec['solver']['max_epochs']!=table['training_updates'] or spec['dataloader']['batch_size']!=table['batch_size']:
                    raise ValueError('Full paper training budget differs')
            identifier=f"fm_spectral_author__{method}__{data['id']}__{data['track']}"
            model_path=ROOT/'configs/models'/f'{identifier}.json'
            if model_path.exists():raise FileExistsError('Model configuration already registered')
            write(model_path,{'schema_version':1,'id':identifier,'task':'unconditional_time_series_generation',
                'family':'pinned_author_generation_pipeline','entrypoint':'iia_benchmark.models.fm_spectral_author:build_author_model',
                'parameters':{'source_root':policy['author_source'],'model_target':spec['model']['target'],'parameters':spec['model']['params']},
                'author_yaml':author_yaml.relative_to(ROOT).as_posix(),'author_yaml_sha256':sha(author_yaml),
                'author_repository':'jw9730/spectral-mean-flow','author_commit':policy['author_commit'],
                'protocol':args.config,'data_case':{'dataset':data['id'],'track':data['track']},
                'reproduction_status':'registered_callable_author_model_full_original_task_execution_pending',
                'citation':policy['citation'],'paper_equivalence_certified':False})
            model_configs.append(model_path.relative_to(ROOT).as_posix())
            for seed in policy['model_seeds']:
                jobid=f"spectral_original__{method}__{data['id']}__{data['track']}__seed{seed}"
                jobs.append({'id':jobid,'model':method,'dataset':data['id'],'data_track':data['track'],'seed':seed,
                    'model_config':model_path.relative_to(ROOT).as_posix(),'author_yaml':author_yaml.relative_to(ROOT).as_posix(),
                    'training_updates':spec['solver']['max_epochs'],'gradient_accumulate_every':spec['solver']['gradient_accumulate_every'],
                    'data_shape':data['shape'],'frozen_samples':data['frozen_samples'],'data_sha256':sha(ROOT/data['frozen_samples']),
                    'reference_path':data['reference_samples'],'reference_sha256':sha(ROOT/data['reference_samples']),
                    'generation_batch_size':policy['generation_batch_size'],'generation_field_chunk_size':policy['generation_field_chunk_size'],
                    'expected_parameters':table.get('expected_parameters') if method=='spectral_flow' else policy['baseline_parameter_counts'].get(data['id']),
                    'output_directory':policy['state_root']+'/jobs/'+jobid,'artifact_directory':policy['artifact_root']+'/'+jobid})
    jobs.sort(key=lambda j:(j['dataset']!='stocks',j['model']!='spectral_flow',j['data_track']!='released_author_data',j['dataset'],j['seed']))
    receipts=[args.config,manifest_path.relative_to(ROOT).as_posix(),lock_path.relative_to(ROOT).as_posix(),claims_path.relative_to(ROOT).as_posix(),
        cache_path.relative_to(ROOT).as_posix(),policy['source_pdf'],'src/iia_benchmark/models/fm_spectral_author.py',
        'scripts/flow_matching/prepare_spectral_original_data.py','scripts/flow_matching/register_spectral_original.py',
        'scripts/flow_matching/run_spectral_original_job.py','scripts/flow_matching/run_spectral_original_queue.py',
        'scripts/flow_matching/preflight_spectral_original.py','configs/reproducibility/spectral_mean_flow_author_py310.requirements.txt',
        'scripts/flow_matching/run_light_controller.py','scripts/flow_matching/run_tsad_queue.py','scripts/flow_matching/heavy_resources.py',
        'scripts/flow_matching/heavy_scheduler_v2.py','scripts/flow_matching/run_grasp_protocol_queue.py','tests/test_fm_spectral_original.py']+model_configs
    receipts += [p.relative_to(ROOT).as_posix() for p in (ROOT/policy['author_source']).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    queue={'schema_version':1,'protocol':args.config,'manifest':manifest_path.relative_to(ROOT).as_posix(),
        'environment_lock':lock_path.relative_to(ROOT).as_posix(),'python':policy['python'],'state_root':policy['state_root'],
        'resource_lock':policy['resource_lock'],'settings':policy['resource_policy'],'jobs':jobs,'registered_cohorts':14,
        'source_receipts':[{'path':p,'sha256':sha(ROOT/p)} for p in dict.fromkeys(receipts)],
        'remaining_original_paper_obligations':policy['remaining_paper_obligations'],'all_original_paper_experiments_complete':False}
    if len(jobs)!=70:raise ValueError('All full five-seed cohorts required')
    write(path,queue)
    print(json.dumps({'registered_jobs':len(jobs),'data_cohorts':7,'model_configs':len(model_configs),'primary_claims':len(claims),'queue':path.relative_to(ROOT).as_posix()}))


if __name__=='__main__':main()
