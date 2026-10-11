"""Register all original Stock/Energy commands and all full independent replays."""
from datetime import datetime,timezone
from pathlib import Path
import json
import psutil

from scripts.flow_matching.run_gtgan_all8_full_native_replay_v1 import ROOT,read,sha,budget,verify,exclusive


def verify_loader_record(dataset,record):
    if any(record.get(key) is not True for key in ['full_original_truth_and_time_row_identity',
            'original_seed56789_whole_row_mask_exact','all_observed_values_exact']):
        raise ValueError('Complete original truth and row masking audit required for every command')
    regular_stock=dataset=='stock' and record['missing_rate']==0
    if record.get('full_original_cubic_coefficients_finite') is not (not regular_stock):
        raise ValueError('Exact original regular-loader or full cubic-coefficient audit required')
    if regular_stock and record['spline_coefficient_shapes']!=[]:
        raise ValueError('Original regular Stock loader has no spline coefficients')
    if not regular_stock and len(record['spline_coefficient_shapes'])!=4:
        raise ValueError('All four original full cubic-coefficient arrays required')
    if dataset=='stock' and record.get('native_original_loader_executed') is not True:
        raise ValueError('Actual full Stock source loader execution required')


def main():
    path='configs/runtime/fm_gtgan_all8_full_native_replay.v1.json'
    queues={'stock':'configs/experiments/fm_gtgan_native_original_stock_queue.v1.json',
        'energy':'configs/experiments/fm_gtgan_native_original_energy_queue.v1.json'}
    audits={'stock':'projects/flow_matching_research/results/2026-10-11/gtgan_native_stock_data_audit_v1/original_stock_loader_audit.json',
        'energy':'projects/flow_matching_research/results/2026-10-11/gtgan_energy_native_audit_v3/native_energy_data_and_model_audit.json'}
    observation='configs/runtime/fm_gtgan_stock_observation_dispatch.v2.json'
    native_states={'stock':read(ROOT/observation)['state_root']+'/status.json',
        'energy':'experiments/runs/fm_gtgan_native_original_energy_v1/status.json'}
    routes=[];owners=[]
    sources={observation,'scripts/flow_matching/run_gtgan_all8_full_native_replay_v1.py',
        'scripts/flow_matching/register_gtgan_all8_full_native_replay_v1.py',
        'tests/test_fm_gtgan_all8_full_native_replay_v1.py',
        'scripts/flow_matching/audit_gtgan_native_models_v1.py',
        'scripts/flow_matching/gtgan_energy_native_common_v1.py',
        'scripts/flow_matching/register_gtgan_native_commands_v1.py',
        'scripts/flow_matching/run_gtgan_energy_native_queue_v1.py',
        'scripts/flow_matching/run_gtgan_energy_native_job_v1.py',
        'scripts/flow_matching/run_spectral_original_queue.py',
        'scripts/flow_matching/run_tsad_queue.py','scripts/flow_matching/heavy_resources.py',
        'scripts/flow_matching/heavy_scheduler_v2.py','scripts/flow_matching/run_grasp_protocol_queue.py',
        'scripts/flow_matching/audit_gtgan_native_stock_data_v1.py',
        'scripts/flow_matching/audit_gtgan_energy_native_v1.py'}
    sources.update(p['path'] for p in read(ROOT/observation)['source_receipts'])
    for dataset,qpath in queues.items():
        queue=read(ROOT/qpath);model=read(ROOT/queue['model_config']);audit=read(ROOT/audits[dataset])
        if not audit['passed'] or audit['performance_result'] or not audit['raw_csv_unchanged']:
            raise ValueError('Full original source-loader truth audit required')
        records=audit['records'] if dataset=='stock' else audit['cases']
        sources.update([qpath,queue['model_config'],queue['environment_lock'],queue['construction_proof'],model['dataset_config'],audits[dataset]])
        sources.update(p['path'] for p in queue['source_receipts'])
        data=read(ROOT/model['dataset_config'])
        if sha(ROOT/data['raw_file'])!=data['raw_sha256']:raise ValueError('Complete original raw data changed')
        sources.add(data['raw_file'])
        state=read(ROOT/native_states[dataset]);owner=psutil.Process(state['pid'])
        if owner.create_time()!=state['create_time']:raise ValueError('Actual original native queue creation time differs')
        fragment='run_gtgan_stock_observation_dispatch_v2' if dataset=='stock' else 'run_gtgan_energy_native_queue_v1'
        if not any(fragment in part for part in owner.cmdline()):raise ValueError('Exact original native owner command required')
        owners.append(dict(dataset=dataset,pid=owner.pid,create_time=owner.create_time(),command=owner.cmdline(),state_path=native_states[dataset]))
        for job in queue['jobs']:
            case=next(c for c in model['cases']+model.get('release_evaluation_cases',[]) if c['id']==job['case_id'])
            record=next(r for r in records if r['missing_rate']==case['missing_rate'])
            verify_loader_record(dataset,record)
            artifact=record['path'] if dataset=='stock' else record['artifact_path']
            digest=record['sha256'] if dataset=='stock' else record['artifact_sha256']
            if sha(artifact)!=digest:raise ValueError('Every complete original source-loader truth array required')
            sources.add(Path(artifact).as_posix())
            preparation=Path(job['runtime_root'])/'runtime_preparation_receipt.json'
            sources.add(preparation.as_posix())
            route=dict(dataset=dataset,original_queue=qpath,canonical_job=job,
                result_path=(Path(job['output_directory'])/'result.json').as_posix(),
                native_completion_receipt=(Path(native_states[dataset]).parent/'receipts'/(job['id']+'.json')).as_posix(),
                settings=queue['settings'],original_truth_audit=dict(manifest=audits[dataset],artifact_path=artifact,
                    artifact_sha256=digest,full_shape=[record['complete_windows'],24,record['variables']]))
            b=budget(route)
            route.update(full_generation_shape=[b['full_generation_windows'],24,b['variables']],
                full_noise_shape=[b['full_generation_windows'],24,b['latent_noise_variables']],
                full_periodic_evaluations=b['full_periodic_evaluations'],full_trained_evaluators=20*b['full_periodic_evaluations'])
            routes.append(route)
    stock=read(ROOT/queues['stock'])
    config=dict(runtime_configuration=path,original_queues=queues,stock_observation_configuration=observation,
        routes=routes,expected_original_native_owners=owners,
        state_root='experiments/runs/fm_gtgan_all8_full_native_replay_v1',
        output_root='F:/aicoding/IIA_Data/experiments/gtgan_all8_full_native_replay_v1/jobs',
        python=stock['python'],settings=stock['settings'],resource_lock=stock['resource_lock'],cpu_threads=2,poll_seconds=10,
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        required_full_generations=60,required_full_trained_evaluators=1200,required_native_network_checkpoint_artifacts=236,
        required_full_generated_windows=717270,equality_gate='bitwise',additional_generator_training_seeds=0,
        citation='Jeon et al. GT-GAN: General Purpose Time Series Synthesis with Generative Adversarial Networks. NeurIPS2022.',
        paper='https://arxiv.org/abs/2210.02040',author_source='https://github.com/Jinsung-Jeon/GT-GAN',
        boundary='Eight complete original native Stock/Energy commands remain: six full-training commands and two '
            'Energy release-weight evaluations, all original seed7777 semantics. All60 periodic full generations, '
            '236 native network checkpoint artifacts and1200 full trained TF model/optimizer checkpoints are required. '
            'Restore whole original network constructors, exact original noise/times/recovery observations/CPU-CUDA RNG '
            'and native modes; call the unchanged full sampler/recovery without chunks or optimizer steps. '
            'Restore complete original TF meta graphs and every model/optimizer variable; execute all saved test feeds '
            'and compare every prediction bitwise. Native prediction graphs cannot reach optimizer operations and '
            'all model/optimizer states must remain unchanged. Full source-loader truth windows including duplicate '
            'counts must agree bitwise under the author float32 conversion, allowing original window shuffling. '
            'Read-only replay adds no training seed or performance observation. Original resources and GPU mutex '
            'unchanged. Registration, cold probes and fixture tests do not prove native numerical equivalence, '
            'TAB certification or full original research completion.',
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(sources)])
    verify(config);exclusive(ROOT/path,config)
    evidence=ROOT/'projects/flow_matching_research/results/2026-10-12/gtgan_all8_full_native_replay_v1'
    exclusive(evidence/'registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        runtime_configuration=path,configuration_sha256=sha(ROOT/path),source_pins=len(config['source_receipts']),
        original_scope=config['original_scope'],all_original_commands=routes,actual_original_native_owners=owners,
        required_full_generations=60,required_full_trained_evaluators=1200,required_network_checkpoint_artifacts=236,
        required_full_generated_windows=717270,equality_gate='bitwise',additional_generator_training_seeds=0,
        native_training_modified_or_stopped=False,numerical_optimizer_steps_during_registration=0,
        full_native_replays_complete=0,new_numeric_results_published=0,all_original_experiments_complete=False))
    print(json.dumps(dict(configuration=path,source_pins=len(config['source_receipts']),full_original_commands=8)))


if __name__=='__main__':main()
