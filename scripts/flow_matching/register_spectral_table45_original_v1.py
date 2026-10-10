"""Freeze every released irregular-stock and physics/stability-loss command."""
import ast
import json
from pathlib import Path
import shlex

from scripts.flow_matching.run_spectral_table45_original_v1 import ROOT, original_args, read, sha


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def original_commands(source):
    for script in ('spectral_flow_pendulum.sh', 'kovae_pendulum.sh', 'spectral_flow_irregular.sh', 'kovae_irregular.sh'):
        for line in (source/'scripts'/script).read_text(encoding='utf-8').splitlines():
            tokens = shlex.split(line, comments=True)
            if tokens:
                if tokens[1] != 'python3' or not tokens[0].startswith('CUDA_VISIBLE_DEVICES='):
                    raise ValueError('Unknown original command shape')
                yield script, tokens[2], tokens[3:], tokens[0]


def main():
    relative = Path('experiments/runs/fm_project_acquisition/sources/spectral_mean_flow/original/src_small')
    source = ROOT/relative
    parent = read(ROOT/'configs/experiments/fm_spectral_mean_flow_original_queue.v2.json')
    artifact_root = Path(parent['jobs'][0]['artifact_directory']).parent.parent/'spectral_table45_original_v1'
    queue_path = 'configs/experiments/fm_spectral_table45_original_queue.v1.json'
    state_root = 'experiments/runs/fm_spectral_table45_original_v1'
    claims = {'spectral_flow':{'regular':[.009,.008], 'interpolate_0.3':[.020,.011], 'interpolate_0.5':[.019,.008],
        'interpolate_0.7':[.015,.007], 'joint_0.3':[.049,.017], 'joint_0.5':[.044,.017], 'joint_0.7':[.138,.137],
        'pendulum_eig0.0':[.0029,.0008], 'pendulum_eig1.0':[.0005,.0004]},
        'kovae':{'regular':[.021,.022], 'interpolate_0.3':[.109,.051], 'interpolate_0.5':[.067,.038],
        'interpolate_0.7':[.049,.052], 'joint_0.3':[.227,.096], 'joint_0.5':[.211,.078], 'joint_0.7':[.187,.075],
        'pendulum_eig0.0':[.0040,.0005], 'pendulum_eig1.0':[.0030,.0004]}}
    sources = {p.relative_to(ROOT).as_posix() for p in source.rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.csv','.pt')}
    sources.update(['scripts/flow_matching/'+name for name in ('run_spectral_table45_original_v1.py',
        'run_spectral_table45_queue_v1.py','register_spectral_table45_original_v1.py',
        'spectral_table2_bootstrap_v1.py','run_spectral_original_queue.py','full_gpu_memory_slot_v2.py',
        'audit_cfm_checkpoint_replay_v1.py','run_cfm_ts_low_memory_queue_v2.py','run_light_controller.py',
        'heavy_resources.py','heavy_scheduler_v2.py','run_grasp_protocol_queue.py','run_tsad_queue.py')])
    sources.update(['tests/test_fm_spectral_table45_original_v1.py', parent['environment_lock'],
        'papers/literature/flow_matching/project_additions/spectral_mean_flow_2510.15366v1.pdf'])
    jobs = []
    for script, entry, arguments, allocation in original_commands(source):
        method = 'kovae' if 'kovae' in entry else 'spectral_flow'
        table = 5 if 'pendulum' in entry else 4
        args = original_args(source/entry, arguments)
        key = 'pendulum_eig'+str(args['w_eig']) if table==5 else (
            'regular' if 'regular' in entry else ('joint_' if 'joint' in entry else 'interpolate_')+str(args['missing_value']))
        data_key = 'pendulum' if table==5 else key
        identity = 'spectral_table'+str(table)+'__'+method+'__'+key.replace('.','p')+'__seed'+str(args['seed'])
        workers = next(n.value.value for n in ast.walk(ast.parse((source/entry).read_text(encoding='utf-8')))
                      if isinstance(n, ast.keyword) and n.arg=='num_workers')
        cfg = dict(id=identity, method=method, table=table, dataset='pendulum' if table==5 else 'stock',
            task='physics_informed_generation' if table==5 else 'irregular_time_series_generation',
            entry=(relative/entry).as_posix(), arguments=arguments, author_arguments=args,
            original_shell=(relative/'scripts'/script).as_posix(), original_allocation=allocation,
            num_workers=workers, data_audit_key=data_key, diagnostic=False,
            metric='cross_correlation' if table==5 else 'discriminative',
            evaluator_repeats=5 if table==5 else 10, evaluator_updates_per_repeat=0 if table==5 else 2000,
            paper_claim=dict(mean=claims[method][key][0], reported_plus_minus=claims[method][key][1],
                spread_definition='Table4 source reports population STD; Table5 source reports t95 half-width over5 subsample evaluations'),
            implementation='scripts.flow_matching.run_spectral_table45_original_v1:run_job',
            reproduction_status='full_released_pipeline_registered_not_completed',
            citations=['https://arxiv.org/abs/2510.15366v1','https://github.com/jw9730/spectral-mean-flow',
                       'https://github.com/azencot-group/KoVAE'],
            boundary='Exact released600-epoch source and arguments. One generator seed10; evaluator repetitions are not training seeds. Original complete-data generation with no heldout generator split, not strict TSAD. CUDA index remapped to sole localGPU; model unchanged.')
        path = 'configs/models/fm_'+identity+'.json'; write_new(ROOT/path, cfg); sources.add(path)
        jobs.append(dict(id=identity, model_config=path, model_config_sha256=sha(ROOT/path),
            artifact_directory=str(artifact_root/identity), output_directory=state_root+'/jobs/'+identity))
    settings = dict(minimum_available_memory_bytes=32*2**30, minimum_commit_headroom_bytes=42*2**30,
        emergency_commit_headroom_bytes=16*2**30, gpu_index=0, gpu_uuid='GPU-b62da557-0167-4b14-25ef-7b6eab536ef9',
        minimum_gpu_free_bytes=14*2**30, emergency_gpu_free_bytes=4*2**30, cpu_threads=2, controller_yield_seconds=35)
    audit_settings = dict(minimum_available_memory_bytes=12*2**30, minimum_commit_headroom_bytes=22*2**30,
        emergency_commit_headroom_bytes=16*2**30, maximum_worker_rss_bytes=2*2**30, maximum_worker_private_bytes=4*2**30)
    queue = dict(schema_version=1, queue_path=queue_path, source_root=relative.as_posix(),
        python=parent['python'], environment_lock=parent['environment_lock'], state_root=state_root,
        resource_lock='experiments/runs/fm_grasp_protocol_gpu.lock', settings=settings,
        data_audit_lock='experiments/runs/fm_readonly_checkpoint_audit_cpu.lock', data_audit_settings=audit_settings,
        data_audit_artifact=str(artifact_root/'full_data_audit'), data_audit_output=state_root+'/preflight/full_data_audit.json',
        source_receipts=[dict(path=p, sha256=sha(ROOT/p)) for p in sorted(sources)], jobs=jobs,
        all_original_experiments_complete=False, full_paper_complete=False,
        remaining_table4_adopted_baselines=['GT-GAN','TimeGAN','RCGAN','C-RNN-GAN','RNN-AR'],
        baseline_obligations='Each adopted baseline still needs0/30/50/70% irregular-to-regular execution; no paper numbers copied as measured results. Table4 joint0% reuses the corresponding regular task, without duplicate training.',
        source_boundaries=['Original irregular cached data contain expected NaNs; full spline coefficients are finite.',
            'Joint cache paths in release do not match loader paths; generate exactly as original in private workspace.',
            'Pendulum uses sourcegravity9.81 and original additional global standardization after MinMax; paper text says9.8 and[0,1].',
            'Windows nested worker initializer is mapped to identical picklable callback; original4 workers retained.',
            'Original cached trusted tensors require weights_only=False under torch2.8. No raw cache overwritten.'])
    write_new(ROOT/queue_path, queue)
    print(json.dumps(dict(queue=queue_path, full_jobs=len(jobs), source_bindings=len(sources), full_epochs=600)))


if __name__ == '__main__':
    main()
