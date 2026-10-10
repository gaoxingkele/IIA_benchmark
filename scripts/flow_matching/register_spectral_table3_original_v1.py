"""Register both full released Table3 methods on both original long-series datasets."""
from pathlib import Path
import json

import yaml

from scripts.flow_matching.run_spectral_table3_original_v1 import ROOT,parse_tsf,sha


def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    source=Path('experiments/runs/fm_project_acquisition/sources/spectral_mean_flow/original/src_large')
    datasets={}
    for name,shape in [('fred_md',[107,728,1]),('nn5_daily',[111,791,1])]:
        raw=source/'data/long_range'/(name+'.tsf');rows=parse_tsf(ROOT/raw)
        assert [len(rows),len(rows[0]),1]==shape
        datasets[name]=dict(raw_path=raw.as_posix(),shape=shape,paper_stated_length=728)
    parent=json.loads((ROOT/'configs/experiments/fm_spectral_table2_original_queue.v1.json').read_text(encoding='utf-8'))
    native_root=Path(parent['jobs'][0]['artifact_directory']).parent.parent/'spectral_table3_original_v1'
    sources={p.relative_to(ROOT).as_posix() for p in (ROOT/source).rglob('*')
             if p.is_file() and p.suffix in ('.py','.yaml','.tsf')}
    sources.update(['scripts/flow_matching/register_spectral_table3_original_v1.py',
        'scripts/flow_matching/run_spectral_table3_original_v1.py',
        'scripts/flow_matching/spectral_table3_bootstrap_v1.py',
        'scripts/flow_matching/run_spectral_table3_queue_v1.py',
        'scripts/flow_matching/spectral_table2_bootstrap_v1.py',
        'scripts/flow_matching/run_spectral_table2_method_gated_queue_v1.py',
        'scripts/flow_matching/run_spectral_original_queue.py',
        'scripts/flow_matching/run_grasp_protocol_queue.py',
        'scripts/flow_matching/run_light_controller.py',
        'scripts/flow_matching/heavy_scheduler_v2.py','scripts/flow_matching/heavy_resources.py',
        'scripts/flow_matching/run_tsad_queue.py',
        'tests/test_fm_spectral_table3_original_v1.py',parent['environment_lock'],
        'scripts/flow_matching/freeze_spectral_table3_environment_v1.py',
        'configs/acquisition/spectral_table3_environment_additions.v1.json',
        'configs/reproducibility/spectral_table3_author_py310.lock.json',
        'papers/literature/flow_matching/project_additions/spectral_mean_flow_2510.15366v1.pdf'])
    claims={'spectral_flow':{'fred_md':[.019,1.338,.753,.030,.006],'nn5_daily':[.006,.950,.257,.539,.196]},
            'imagentime':{'fred_md':[.022,.755,.343,.034,.020],'nn5_daily':[.009,.560,.174,.584,.188]}}
    jobs=[]
    for method in ('spectral_flow','imagentime'):
        for dataset,spec in datasets.items():
            identity='spectral_table3__'+method+'__'+dataset+'__seed42'
            original=source/'configs'/method/(dataset+'.yaml')
            hyper=yaml.safe_load((ROOT/original).read_text(encoding='utf-8'))
            assert hyper['epochs']==1000 and hyper['batch_size']==32
            config=dict(id=identity,method=method,dataset=dataset,seed=42,evaluation_seed=0,
                diagnostic=False,device='cuda',author_config=original.as_posix(),author_config_sha256=sha(ROOT/original),
                author_hyperparameters=hyper,raw_path=spec['raw_path'],raw_sha256=sha(ROOT/spec['raw_path']),
                training_entry=(source/('main.py' if method=='spectral_flow' else 'main_imagentime.py')).as_posix(),
                evaluation_entry=(source/('eval.py' if method=='spectral_flow' else 'eval_imagentime.py')).as_posix(),
                implementation='scripts.flow_matching.run_spectral_table3_original_v1:run_job',
                reproduction_status='full_original_pipeline_registered_preflight_and_training_pending',
                citations=['https://arxiv.org/abs/2510.15366v1','https://github.com/azencot-group/ImagenTime'],
                source_status='Pinned released Spectral Mean Flow Table3 implementation, including its ImagenTime reproduction; exact historical author equivalence not certified.',
                paper_claims=dict(marginal_score_mean=claims[method][dataset][0],clf_score_mean=claims[method][dataset][1],
                    clf_score_std=claims[method][dataset][2],predictive_score_mean=claims[method][dataset][3],predictive_score_std=claims[method][dataset][4]),
                boundary='One released generator seed42; STD from10 S4 evaluator fits. Retain separate evaluation seed0, original full-series normalization, test marginal checkpoint selection and validation timing. NN5 release has791 timestamps whereas paper text says728; do not crop.')
            path=Path('configs/models')/('fm_'+identity+'.json');write_new(ROOT/path,config);sources.add(path.as_posix())
            jobs.append(dict(id=identity,model_config=path.as_posix(),model_config_sha256=sha(ROOT/path),
                artifact_directory=str(native_root/identity),preflight_artifact_directory=str(native_root/('preflight_'+identity)),
                output_directory='experiments/runs/fm_spectral_table3_original_v1/jobs/'+identity))
    settings=dict(minimum_available_memory_bytes=32*2**30,minimum_commit_headroom_bytes=42*2**30,
        emergency_commit_headroom_bytes=16*2**30,gpu_index=0,gpu_uuid='GPU-b62da557-0167-4b14-25ef-7b6eab536ef9',
        minimum_gpu_free_bytes=14*2**30,emergency_gpu_free_bytes=4*2**30,controller_yield_seconds=35)
    queue=dict(schema_version=1,queue_path='configs/experiments/fm_spectral_table3_original_queue.v1.json',
        source_root=source.as_posix(),python='.venv/spectral-table3-author-py310/Scripts/python.exe',
        environment_lock='configs/reproducibility/spectral_table3_author_py310.lock.json',datasets=datasets,
        cpu_data_audit='experiments/runs/fm_spectral_table3_original_v1/preflight/cpu_data_v2/proof.json',
        state_root='experiments/runs/fm_spectral_table3_original_v1',
        resource_lock='experiments/runs/fm_grasp_protocol_gpu.lock',cpu_reservation_lock='experiments/runs/fm_heavy_recovery_cpu.lock',
        settings=settings,source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(sources)],jobs=jobs,
        remaining_original_baselines=['LS4','SaShiMi-AR'],all_original_experiments_complete=False,benchmark_smoke=False,
        boundary='Full native released Table3 pipelines, not short-term generation, forecasting or TSAD surrogates. Four full1000-epoch jobs; full test generation and ten100-epoch S4 classifier/predictor repeats per validation/final evaluation. Source-only registration and capacity diagnostics are not performance.')
    write_new(ROOT/queue['queue_path'],queue)
    print(json.dumps(dict(full_jobs=len(jobs),source_bindings=len(sources),queue=queue['queue_path'],raw_shapes={k:v['shape'] for k,v in datasets.items()})))


if __name__=='__main__':main()
