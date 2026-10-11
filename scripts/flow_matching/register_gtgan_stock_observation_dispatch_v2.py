"""Freeze observation of all four complete native GT-GAN stock commands."""
from datetime import datetime, timezone
from pathlib import Path
import json

import psutil

from scripts.flow_matching.gtgan_stock_observation_runtime_v2 import ROOT, read, sha, extended_budget
from scripts.flow_matching.run_gtgan_stock_observation_dispatch_v2 import verify, all_attempts_unstarted, idle_parent_phase, exclusive


def main():
    path = 'configs/runtime/fm_gtgan_stock_observation_dispatch.v2.json'
    queue_path = 'configs/experiments/fm_gtgan_native_original_stock_queue.v1.json'
    queue = read(ROOT / queue_path)
    model = read(ROOT / queue['model_config'])
    state_path = queue['state_root'] + '/status.json'
    state = read(ROOT / state_path)
    owner = psutil.Process(state['pid'])
    if (owner.create_time() != state['create_time'] or state['pid'] != 29268
            or state['create_time'] != 1791666646.7949967
            or owner.cmdline()[3:] != ['-u','-m','scripts.flow_matching.run_gtgan_native_queue_v1','--queue',queue_path]):
        raise ValueError('Actual original stock parent identity changed; inspect before registration')
    descendants = [dict(pid=p.pid,name=p.name(),exe=p.exe()) for p in owner.children(recursive=True)]
    if not idle_parent_phase(state,descendants,all_attempts_unstarted(queue)):
        raise ValueError('Only the four completely unstarted stock tasks behind an idle GPU parent may be handed off')
    sources = {queue_path,queue['environment_lock'],queue['model_config'],queue['construction_proof'],
        model['dataset_config'],'scripts/flow_matching/gtgan_stock_observation_runtime_v2.py',
        'scripts/flow_matching/run_gtgan_stock_observation_dispatch_v2.py',
        'scripts/flow_matching/register_gtgan_stock_observation_dispatch_v2.py',
        'scripts/flow_matching/gtgan_energy_metric_receipts_v1.py',
        'scripts/flow_matching/run_timegan_corrected_observer_dispatch_v2.py',
        'scripts/flow_matching/timegan_predictive_observer_v2.py',
        'scripts/flow_matching/timegan_native_common_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/run_spectral_original_queue.py',
        'scripts/flow_matching/run_light_controller.py',
        'scripts/flow_matching/full_gpu_memory_slot_v2.py',
        'tests/test_fm_gtgan_stock_observation_v2.py'}
    sources.update(p['path'] for p in queue['source_receipts'])
    data = read(ROOT / model['dataset_config'])
    if sha(ROOT / data['raw_file']) != data['raw_sha256']:
        raise ValueError('Complete original stock raw data changed')
    sources.add(data['raw_file'])
    runtime_receipts = []
    for job in queue['jobs']:
        runtime = Path(job['runtime_root'])
        receipt = runtime / 'runtime_preparation_receipt.json'
        proof = read(receipt)
        if len(proof['complete_copied_assets']) != 245:
            raise ValueError('Every original full native runtime asset required')
        repairs = {r['path']:r['repaired_sha256'] for r in proof['runtime_repairs']}
        for asset in proof['complete_copied_assets']:
            if sha(runtime / asset['path']) != repairs.get(asset['path'],asset['original_sha256']):
                raise ValueError('Unstarted original runtime asset changed: ' + asset['path'])
            sources.add((Path(model['source_root']) / asset['path']).as_posix())
        sources.add(receipt.as_posix())
        runtime_receipts.append(dict(job_id=job['id'],path=receipt.as_posix(),sha256=sha(receipt),full_copied_assets=245))
    budgets = [extended_budget(c) for c in model['cases']]
    config = dict(runtime_configuration=path,original_queue=queue_path,
        state_root='experiments/runs/fm_gtgan_stock_observation_dispatch_v2',
        previous_owner_route=dict(previous_owner=dict(pid=owner.pid,create_time=owner.create_time(),command=owner.cmdline()),
            previous_state_path=state_path),settings=queue['settings'],resource_lock=queue['resource_lock'],
        cpu_threads=queue['cpu_threads'],original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        required_trained_evaluator_receipts=sum(b['full_periodic_evaluations']*20 for b in budgets),
        required_sampler_snapshots=sum(b['full_periodic_evaluations'] for b in budgets),
        required_native_network_checkpoints=sum(b['full_periodic_evaluations']*4 for b in budgets),
        runtime_preparation_receipts=runtime_receipts,
        native_latent_noise_shape=[3662,24,24],native_real_and_generated_shape=[3662,24,6],
        citation='Jeon et al. GT-GAN: General Purpose Time Series Synthesis with Generative Adversarial Networks. NeurIPS2022.',
        paper='https://arxiv.org/abs/2210.02040',author_source=data['source_repository'],
        boundary='All four full original stock commands/seed7777 remain. Regular released pretrain skip and irregular '
            '10000 embedding/2000 joint updates, released70percent reload, sym12async solver and full3662-window '
            'sampling preserved. Original30/50percent parameter extensions remain explicitly qualified. '
            'Only sampler noise/times/native CPU-CUDA RNG/modes and all2000/5000-update evaluator weights, '
            'optimizer states, full test feeds/targets/predictions are observed. Full29 periodic generations, '
            '116 native network checkpoints and580 trained critics required. Evaluator repeats are not new generator '
            'training seeds. Source equivalence/TAB certification/independent native replay/performance/full research '
            'completion are not claimed by registration, cold probes or fixture tests.',
        source_receipts=[dict(path=p,sha256=sha(ROOT / p)) for p in sorted(sources)])
    verify(config)
    if owner.create_time()!=state['create_time'] or not all_attempts_unstarted(queue):
        raise ValueError('Original native owner or fresh job state changed during verification')
    exclusive(ROOT / path,config)
    evidence = ROOT / 'projects/flow_matching_research/results/2026-10-12/gtgan_stock_observation_v2'
    exclusive(evidence/'registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        runtime_configuration=path,configuration_sha256=sha(ROOT/path),source_pins=len(config['source_receipts']),
        previous_owner=config['previous_owner_route']['previous_owner'],original_scope=config['original_scope'],
        all_four_original_stock_jobs=queue['jobs'],full_budgets=budgets,
        full_periodic_evaluations=29,full_native_network_checkpoints=116,full_trained_evaluator_instances=580,
        full_native_latent_noise_shape=[3662,24,24],full_native_generation_shape=[3662,24,6],
        author_algorithm_arguments_sources_and_resource_safeguards_unchanged=True,
        full_original_runtime_assets_verified_per_case=245,training_executed_during_registration=False,
        active_native_workers_stopped=False,new_numeric_results_published=0,independent_native_replay_complete=False,
        all_original_experiments_complete=False))
    print(json.dumps(dict(configuration=path,source_pins=len(config['source_receipts']),original_stock_jobs=4)))


if __name__=='__main__':
    main()
