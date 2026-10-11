"""Replay the exact registered full or reconstructed ablation CrossAD model."""
import argparse
import json
import os
import subprocess

import psutil

from scripts.flow_matching import replay_crossad_full_native_model_v1 as original
from scripts.flow_matching.replay_crossad_full_native_model_v2 import restore_original_profile_buffers

ROOT, read, digest = original.ROOT, original.read, original.digest


def select_factory(job, full_factory, ablation_factory, checkpointing):
    if 'ablation_row' in job:
        if (job['mode'] != 'fresh_train' or not job['reconstructed_ablation']
                or job['ablation_row'] not in range(1,6)
                or job['author_recipe'] != 'paper_reconstructed_table3_row'+str(job['ablation_row'])
                or job['runner'] != 'scripts/flow_matching/run_crossad_ablation_v2_job.py'):
            raise ValueError('Use the exact frozen reconstructed ablation, without claiming author architecture equivalence')
        return lambda runtime, parameters: checkpointing(ablation_factory(runtime, parameters, job['ablation_row']))
    if job['reconstructed_ablation'] or job['mode'] not in ('fresh_train', 'release_checkpoint'):
        raise ValueError('Original full model and release evaluation must remain separate from reconstructed ablations')
    return full_factory


def worker(config, path):
    import iia_benchmark.models.fm_crossad_memory_v2 as native
    from iia_benchmark.models.fm_crossad_ablation import build_ablation
    queue = read(config['queue'])
    job = next(j for j in queue['jobs'] if j['id'] == config['job_id'])
    factory = select_factory(job, native.build_memory_model, build_ablation, native.install_activation_checkpointing)
    restored = []
    def faithful_factory(*args, **kwargs):
        model = factory(*args, **kwargs)
        def restore(module, state, prefix, metadata, strict, missing, unexpected, errors):
            if prefix:
                raise ValueError('Restore only the original root state dictionary metadata')
            restored.extend(restore_original_profile_buffers(module, state))
        model.register_load_state_dict_pre_hook(restore)
        return model
    native.build_memory_model = faithful_factory
    original.worker(config, path)
    proof = read(ROOT/config['artifact_directory']/'full_native_model_replay.json')
    proof.update(original_full_state_dictionary_loaded_strictly=True,
        original_THOP_profile_buffers_restored_without_dropping_algorithm_weights=restored,
        registered_author_recipe=job['author_recipe'], registered_ablation_row=job.get('ablation_row'),
        reconstructed_ablation=job['reconstructed_ablation'], unpublished_author_ablation_equivalence_certified=False,
        required_original_cohort_training_seeds=0 if job['mode']=='release_checkpoint' else 6,
        completed_training_seed_slots=0 if job['mode']=='release_checkpoint' else 1,
        fixed_release_checkpoint_evaluation=job['mode']=='release_checkpoint')
    if job['mode']=='release_checkpoint':
        proof.pop('required_original_main_training_seeds')
        proof.update(original_release_checkpoint=job['release_checkpoint'],
            original_release_checkpoint_sha256=digest(job['release_checkpoint']))
    (ROOT/config['artifact_directory']/'full_native_model_replay.json').write_text(
        json.dumps(proof,indent=2)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    config = read(args.config)
    for pin in config['source_receipts']:
        if digest(pin['path']) != pin['sha256']:
            raise ValueError('Frozen registered full-model/ablation replay source changed')
    if args.worker:
        return worker(config, args.config)
    api = original.resource_api()
    base = ROOT/config['state_root']
    base.mkdir(parents=True, exist_ok=True)
    lane = api['exclusive_lock'](base/'controller.lock')
    if lane is None:
        raise RuntimeError('This exact complete registered model replay already has an owner')
    state = dict(pid=os.getpid(),create_time=psutil.Process().create_time(),state='starting',active_pid=None)
    def update(values):
        state.update(values)
        (base/'status.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    try:
        with api['resource_slot'](ROOT/config['resource_lock'], config['settings'], update):
            env = dict(os.environ, PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',
                PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS=str(config['cpu_threads']),
                MKL_NUM_THREADS=str(config['cpu_threads']),OPENBLAS_NUM_THREADS=str(config['cpu_threads']))
            with (base/'stdout.log').open('xb') as out,(base/'stderr.log').open('xb') as err:
                child=subprocess.Popen([config['python'],'-X','utf8','-u','-m',
                    'scripts.flow_matching.replay_crossad_registered_model_v3','--config',args.config,'--worker'],
                    cwd=ROOT,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            handle=psutil.Process(child.pid)
            update(dict(active_pid=child.pid,active_create_time=handle.create_time(),actual_command=handle.cmdline()))
            code,usage=original.capped_wait(child,config,api,update)
        path=ROOT/config['artifact_directory']/'full_native_model_replay.json'
        passed=bool(code==0 and not usage['resource_guard_aborted'] and path.exists()
            and read(path).get('original_full_state_dictionary_loaded_strictly') is True)
        receipt=dict(exit_code=code,full_native_model_replay_passed=passed,**usage)
        (base/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        update(dict(state='completed' if passed else 'failed_preserved',active_pid=None,**receipt))
        if not passed:
            raise RuntimeError('Full registered model replay failed; preserve its entire original attempt')
    finally:
        lane.close()


if __name__=='__main__':
    main()
