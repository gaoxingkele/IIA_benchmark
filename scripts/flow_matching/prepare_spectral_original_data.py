"""Run pinned author's original loaders and preserve frozen generation arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def extract_member(archive, member, base):
    destination=base/member
    if not destination.resolve().is_relative_to(base.resolve()):
        raise ValueError('Archive member escapes the registered root')
    content=archive.read(member)
    expected=hashlib.sha256(content).hexdigest()
    if destination.exists():
        if sha(destination)!=expected:
            raise ValueError('Existing raw data differs; preserve it')
    else:
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(content)
    return destination


def resume_case(destination, bindings):
    """Reuse a verified completed case; preserve any incomplete preparation."""
    if not destination.exists():
        return None
    audit=destination/'data_audit.json'
    if not audit.exists():
        raise FileExistsError('Incomplete preparation preserved: '+str(destination))
    record=read(audit)
    if record.get('preparation_bindings')!=bindings:
        raise ValueError('Completed preprocessing bindings differ; preserve '+str(destination))
    for artifact in record['artifacts']:
        if sha(ROOT/artifact['path'])!=artifact['sha256']:
            raise ValueError('Completed preprocessing bytes differ; preserve '+str(destination))
    return record


def paper_sines(seed, num=10000, window=24, dim=5):
    import numpy as np
    rng=np.random.RandomState(seed)
    draws=rng.uniform(size=(num,dim,2))
    frequency=draws[:,:,0]; phase=(draws[:,:,1]*2-1)*np.pi
    return np.sin(2*np.pi*frequency[:,:,None]*np.arange(window)+phase[:,:,None]).transpose(0,2,1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/reproducibility/spectral_mean_flow_original_protocol.v2.json')
    args=parser.parse_args(); policy=read(ROOT/args.config)
    base=ROOT/policy['data_output']; manifest=base/'manifest.json'
    if manifest.exists():
        existing=read(manifest)
        if existing['protocol_sha256']!=sha(ROOT/args.config):
            raise ValueError('Original preprocessing configuration differs')
        for dataset in existing['datasets']:
            for artifact in dataset['artifacts']:
                if sha(ROOT/artifact['path'])!=artifact['sha256']:
                    raise ValueError('Frozen author data changed')
        print(json.dumps({'verified_data_cohorts':len(existing['datasets'])}));return
    if sha(ROOT/policy['data_archive'])!=policy['data_archive_sha256']:
        raise ValueError('Registered archive identity differs')
    import numpy as np
    import yaml
    sys.path.insert(0,str(ROOT/policy['author_source']))
    from utils.io_utils import instantiate_from_config
    members={'stocks':'datasets/stock_data.csv','etth':'datasets/ETTh.csv',
             'energy':'datasets/energy_data.csv','fmri':'datasets/fMRI/sim4.mat'}
    with zipfile.ZipFile(ROOT/policy['data_archive']) as archive:
        paths={name:extract_member(archive,member,base/'raw') for name,member in members.items()}
    datasets=[]
    for dataset in policy['table1']:
        destination=base/'released_author_data'/dataset
        configuration=ROOT/policy['author_source']/'configs/spectral_flow'/f'{dataset}.yaml'
        bindings={'protocol_sha256':sha(ROOT/args.config),'author_yaml_sha256':sha(configuration),
            'preparation_source_sha256':sha(Path(__file__)),'data_archive_sha256':policy['data_archive_sha256']}
        reused=resume_case(destination,bindings)
        if reused is not None:
            datasets.append(reused);print(json.dumps({'verified_existing_case':dataset}),flush=True);continue
        config=yaml.safe_load(configuration.read_text(encoding='utf-8'))
        spec=config['dataloader']['train_dataset']
        spec['params']['output_dir']=str(destination)
        if dataset in paths:
            spec['params']['data_root']=str(paths[dataset].parent if dataset=='fmri' else paths[dataset])
        original=instantiate_from_config(spec)
        samples=original.samples
        if samples.ndim!=3 or samples.shape[1:]!=(24,policy['table1'][dataset]['features']) or not np.isfinite(samples).all():
            raise ValueError('Full author loader produced invalid dimensions or values')
        with (destination/'frozen_training_samples.npy').open('xb') as stream:
            np.save(stream,samples,allow_pickle=False)
        files=[p for p in destination.rglob('*') if p.is_file()]
        artifacts=[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in files]
        dataset_record={'id':dataset,'track':'released_author_data','shape':list(samples.shape),
            'data_seed':policy['data_seed'],'group_unit':'original complete simulated trajectory' if dataset in ['sines','mujoco'] else 'stride-one windows from the complete source series',
            'frozen_samples':(destination/'frozen_training_samples.npy').relative_to(ROOT).as_posix(),
            'reference_samples':next(a['path'] for a in artifacts if a['path'].endswith('_norm_truth_24_train.npy') or a['path'].endswith('sine_ground_truth_24_train.npy')),
            'artifacts':artifacts,'scaler_fit':'full source series or all simulated trajectories; author distribution reconstruction protocol',
            'window_ids':'all original author-loader windows in original order; no held-out trajectory claim',
            'raw_source':paths[dataset].relative_to(ROOT).as_posix() if dataset in paths else 'pinned author simulation loader',
            'author_yaml_sha256':sha(configuration),'preparation_bindings':bindings}
        write(destination/'data_audit.json',dataset_record);datasets.append(dataset_record)
        print(json.dumps({'prepared':dataset,'shape':list(samples.shape)}),flush=True)
    case=base/'paper_sines_distribution'/'sines'
    bindings={'protocol_sha256':sha(ROOT/args.config),'preparation_source_sha256':sha(Path(__file__))}
    reused=resume_case(case,bindings)
    if reused is None:
        case.mkdir(parents=True)
        samples=paper_sines(policy['data_seed'],**{k:policy['paper_sines_distribution'][k] for k in ('num','window','dim')})
        for name,array in [('frozen_training_samples.npy',samples),('sine_ground_truth_24_train.npy',(samples+1)*.5)]:
            with (case/name).open('xb') as stream:np.save(stream,array,allow_pickle=False)
        reused={'id':'sines','track':'paper_sines_distribution','shape':list(samples.shape),'data_seed':policy['data_seed'],
        'frozen_samples':(case/'frozen_training_samples.npy').relative_to(ROOT).as_posix(),
        'reference_samples':(case/'sine_ground_truth_24_train.npy').relative_to(ROOT).as_posix(),
        'artifacts':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in case.iterdir() if p.is_file()],
        'group_unit':'complete simulated trajectory','raw_source':'paper p25 formula','author_equivalence_certified':False,
        'preparation_bindings':bindings}
        write(case/'data_audit.json',reused)
    datasets.append(reused)
    write(manifest,{'protocol':args.config,'protocol_sha256':sha(ROOT/args.config),'preparation_source_sha256':sha(Path(__file__)),
        'source_archive_sha256':policy['data_archive_sha256'],'source_archive_preserved':True,
        'datasets':datasets,'original_task':'unconditional_time_series_generation','benchmark_smoke':False,
        'all_original_paper_experiments_complete':False})
    print(json.dumps({'prepared_data_cohorts':len(datasets)}))


if __name__=='__main__':
    main()
