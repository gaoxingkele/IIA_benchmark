"""Register exact released Table2 CLI pipelines without substituting smaller models."""
import argparse
import ast
from collections import defaultdict
import csv
import json
from pathlib import Path
import shlex
import struct

from scripts.flow_matching.freeze_spectral_table2_environment_v1 import ROOT, sha


def option(arguments, flag, default=None):
    return arguments[arguments.index(flag)+1] if flag in arguments else default


def source_pipelines(source):
    pipelines = defaultdict(list)
    for method, script in [('spectral_flow','spectral_flow.sh'),('sdformer_ar','sdformer.sh')]:
        for line in (source/'scripts'/script).read_text(encoding='utf-8').splitlines():
            if not line.strip().startswith('python3 '):
                continue
            command = shlex.split(line)
            entry, arguments = command[1], command[2:]
            dataset = option(arguments,'--dataname') or Path(option(arguments,'--dir')).parts[-2]
            stage = 'evaluation' if entry.startswith('eval') else 'sampling' if '--if-test' in arguments else 'vq_training' if entry=='train_vq.py' else 'training'
            updates = 0 if stage in ('evaluation','sampling') else int(option(arguments,'--total-iter'))
            warmup = 999 if stage=='vq_training' else 0
            interval = 2000 if stage=='vq_training' else int(option(arguments,'--eval-iter',0))
            pipelines[method,dataset].append(dict(stage=stage,entry=entry,original_arguments=arguments,
                optimizer_updates=updates+warmup, main_optimizer_updates=updates, warmup_updates=warmup,
                periodic_eval_interval=interval, discriminator_repeats=3 if stage=='vq_training' else 5,
                model_keys=['net'] if stage=='vq_training' else ['net','trans_encoder'] if method=='sdformer_ar' else ['model','ema'],
                selected_checkpoint=str(Path(option(arguments,'--out-dir',''))/option(arguments,'--exp-name','')/'net_best_ds.pth') if updates else None))
    if len(pipelines)!=6 or any(len(stages)!=(4 if key[0]=='sdformer_ar' else 3) for key,stages in pipelines.items()):
        raise ValueError('Released six full native pipelines missing')
    return pipelines


def npy_shape(path):
    with path.open('rb') as stream:
        if stream.read(6)!=b'\x93NUMPY':raise ValueError('Not an NPY array')
        major, minor=stream.read(2)
        size=struct.unpack('<H' if major==1 else '<I',stream.read(2 if major==1 else 4))[0]
        return list(ast.literal_eval(stream.read(size).decode('latin1'))['shape'])


def frozen_write(path, data):
    payload=json.dumps(data,ensure_ascii=False,indent=2)+'\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_text(encoding='utf-8')!=payload:raise ValueError('Existing registered config differs: '+str(path))
    else:path.write_text(payload,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/reproducibility/spectral_table2_original_protocol.v1.json')
    args=parser.parse_args()
    source=ROOT/'experiments/runs/fm_project_acquisition/sources/spectral_mean_flow/original/src_medium'
    assets=json.loads((ROOT/'configs/data/spectral_additional_original_assets.v1.json').read_text(encoding='utf-8'))
    items={x['id']:x for x in assets['assets']}
    data={}
    for name,asset_id in [('sine','table2_sines'),('stock','table2_stocks'),('mujoco','table2_mujoco')]:
        asset=items[asset_id]; path=ROOT/asset['destination']
        if sha(path)!=asset['sha256']:raise ValueError('Registered original data changed')
        if path.suffix=='.npy':shape=npy_shape(path)
        else:
            with path.open(encoding='utf-8-sig',newline='') as stream:
                rows=csv.reader(stream);columns=len(next(rows));count=sum(1 for _ in rows)
            shape=[count-24+1,24,columns]
        data[name]=dict(path=asset['destination'],sha256=asset['sha256'],workspace_path='dataset/'+path.name,shape=shape,
            preprocessing='released_NPY_unchanged' if name!='stock' else 'author_TSDataset_full_series_feature_minmax_no_epsilon_stride1')
    protocol=dict(schema_version=1,paper=assets['paper'],table=2,author_source=source.relative_to(ROOT).as_posix(),
        author_repository='https://github.com/jw9730/spectral-mean-flow',author_commit='81e49338b10b21262b8069703cac3639aa0d43aa',
        environment_lock='configs/reproducibility/spectral_table2_author_py310.lock.json',
        data=data,seeds=[42,1103,1104,1105,1106],metric_repeats=5,metrics=['context_fid','discriminative','predictive'],
        metric_seed=0,source_num_workers=8,source_cli_changes=['GPU device remap to0','training seed specified per registered job'],
        final_checkpoint_selection='author_periodic_discriminative_score_best_checkpoint',
        sampling='full_original_sampler_unmodified_no_chunking',
        windows_fix='execute byte-identical entry in separate namespace while retaining guarded outer multiprocessing __main__',
        pending_original_table2_methods={'sdformer_m':'Source exists; released full dataset-specific CLI recipe absent in src_medium/scripts; do not infer AR hyperparameters',
            'imagentime':'Released source/configs in src_large; native execution pipeline remains to implement'},
        additional_original_obligations='Full Tables3/4/5, tensor-network/memory and physics figures remain in original frontier; Table1 remains separate live70-job queue',
        source_rng_boundary='Author RNG setup is retained exactly. AR only seeds Torch; TF evaluators reset graphs without reseeding. Job seed repeats alone do not certify all RNGs or author equivalence.')
    frozen_write(ROOT/args.config,protocol)
    pipelines=source_pipelines(source)
    jobs=[];model_configs=[]
    for (method,dataset),stages in pipelines.items():
        model_path='configs/models/fm_spectral_table2__'+method+'__'+dataset+'.json'
        model=dict(schema_version=1,id=Path(model_path).stem,task='unconditional_time_series_generation',
            family='pinned_author_multistage_generation_pipeline',
            entrypoint='scripts.flow_matching.run_spectral_table2_original_v1:run_job',
            protocol=args.config,author_commit=protocol['author_commit'],dataset=dataset,
            stages=stages,reproduction_status='full_released_native_pipeline_registered_preflight_pending',
            citation=protocol['paper']['citation'],paper_equivalence_certified=False)
        frozen_write(ROOT/model_path,model);model_configs.append(model_path)
        for seed in protocol['seeds']:
            identity=f'spectral_table2__{method}__{dataset}__seed{seed}'
            job=dict(id=identity,method=method,dataset=dataset,seed=seed,model_config=model_path,
                stages=stages,data=data[dataset],output_directory='experiments/runs/fm_spectral_table2_original_v1/jobs/'+identity,
                artifact_directory=str(Path('F:/aicoding/IIA_Data/experiments/spectral_table2_original_v1')/identity))
            jobs.append(job)
    sources=[ROOT/args.config,ROOT/protocol['environment_lock'],ROOT/'configs/data/spectral_additional_original_assets.v1.json',
        ROOT/'configs/acquisition/spectral_table2_environment_additions.v1.json']
    sources += [p for p in source.rglob('*') if p.is_file() and p.suffix in ('.py','.sh') and '__pycache__' not in p.parts]
    sources += [ROOT/p for p in model_configs]
    sources += [ROOT/'scripts/flow_matching'/p for p in ('spectral_table2_bootstrap_v1.py','run_spectral_table2_original_v1.py','run_spectral_table2_queue_v1.py')]
    queue=dict(schema_version=1,protocol=args.config,python='.venv/spectral-table2-author-py310/Scripts/python.exe',
        environment_lock=protocol['environment_lock'],state_root='experiments/runs/fm_spectral_table2_original_v1',
        resource_lock='experiments/runs/fm_grasp_protocol_gpu.lock',
        cpu_preflight_lock='experiments/runs/fm_cfm_ts_original_cpu.lock',
        settings=dict(gpu_index=0,minimum_gpu_free_bytes=14*1024**3,minimum_available_memory_bytes=18*1024**3,
            minimum_commit_headroom_bytes=28*1024**3,emergency_commit_headroom_bytes=16*1024**3),
        cpu_preflight_settings=dict(minimum_available_memory_bytes=10*1024**3,
            minimum_commit_headroom_bytes=20*1024**3,emergency_commit_headroom_bytes=16*1024**3),
        source_receipts=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in sources],jobs=jobs,
        benchmark_smoke=False,full_original_scope_complete=False)
    frozen_write(ROOT/'configs/experiments/fm_spectral_table2_original_queue.v1.json',queue)
    print(json.dumps(dict(full_original_jobs=len(jobs),full_method_dataset_pipelines=len(pipelines),data=data,protocol=args.config)))


if __name__=='__main__':main()
