"""Register full re-inference of the completed canonical source-runtime PSM1107 model."""
import copy
from datetime import datetime,timezone
import json
from pathlib import Path

from scripts.flow_matching.run_pi_source1107_completed_full_replay_v1 import ROOT,read,digest,verify,IDS,BUDGET,CANONICAL
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json


def aligned_review_metadata(template):
    result=copy.deepcopy(template)
    result['boundary']='Complete original '+CANONICAL+' checkpoint replay, all15 full saved/replayed streams and independent metric review. Original full1140/six-seed scope, numerical settings, resource guards and tolerances remain unchanged. Historical training resource guard receipt is missing and remains unproven. No new training seed or author/strictTAB certificate.'
    return result


def main():
    native_path='configs/reproducibility/fm_pi_original_source_full_checkpoint_replay.2026-10-11_v1.json'
    point_path='configs/reproducibility/fm_pi_original_source_recovery_completion.2026-10-11_v1.json'
    review_path='configs/reproducibility/fm_pi_original_completed_full_replay_review.2026-10-11_v1.json'
    native,point,review=read(native_path),read(point_path),read(review_path)
    queue_path='configs/experiments/fm_pi_transformer_author_queue.v1.json';queue=read(queue_path)
    original_review_template=review_path
    review_path='configs/reproducibility/fm_pi_source1107_completed_full_replay_children_v1/readonly_review_template.json'
    exclusive_json(ROOT/review_path,aligned_review_metadata(review))
    review=read(review_path)
    config=dict(python=review['python'],original_queue=queue_path,required_original_jobs=1140,
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        native_template=native_path,review_template=review_path,poll_seconds=5,
        state_root='experiments/runs/fm_pi_source1107_completed_full_replay_v1',
        child_state_root='experiments/runs/fm_pi_source1107_completed_full_replay_children_v1',
        configuration_root='configs/reproducibility/fm_pi_source1107_completed_full_replay_children_v1',
        proof_root='projects/flow_matching_research/results/2026-10-15/pi_source1107_completed_full_replay_v1/completed_models',
        artifact_root='F:/aicoding/IIA_Data/experiments/pi_source1107_completed_full_replay_v1',cases=[], **BUDGET)
    binding_path=config['configuration_root']+'/readonly_original_canonical_bindings.json'
    cases=[];sources={queue_path,native_path,point_path,review_path,original_review_template,
        'scripts/flow_matching/run_pi_source1107_completed_full_replay_v1.py',
        'scripts/flow_matching/register_pi_source1107_completed_full_replay_v1.py',
        'tests/test_fm_pi_source1107_completed_full_replay_v1.py',
        'scripts/flow_matching/verify_pi_original_full_replay_completion_v1.py',
        'scripts/flow_matching/run_pi_causal_completed_pair_replay_v1.py',
        'scripts/flow_matching/register_pi_causal_completed_pair_replay_v1.py',
        'tests/test_fm_pi_causal_completed_pair_replay_v1.py'}
    for template in (native,point,review):sources.update(pin['path'] for pin in template['source_receipts'])
    for identity in IDS:
        job=next(j for j in queue['jobs'] if j['id']==identity)
        cases.append(dict(kind='readonly_original_canonical_binding',original_queue=queue_path,
            original_queue_sha256=digest(queue_path),original_id=identity,original_job=copy.deepcopy(job),
            job=copy.deepcopy(job),settings=copy.deepcopy(queue['settings'])))
        sources.update(pin['path'] for pin in job['frozen_files'])
        result_path=job['output_directory']+'/result.json';result=read(result_path)
        sources.add(result_path);sources.update(a['path'] for a in result['artifacts'])
    exclusive_json(ROOT/binding_path,dict(jobs=cases,
        boundary='Readonly bindings to exact original canonical jobs and output directories, not new recovery/training jobs. Original full1140 registry unchanged.'))
    sources.add(binding_path)
    pins=[dict(path=p,sha256=digest(p)) for p in sorted(sources)]
    for identity in IDS:
        point_config=copy.deepcopy(point)
        point_config.update(recovery_queue=binding_path,recovery_job_id=identity,python=review['python'],
            settings=copy.deepcopy(review['settings']),resource_lock=review['resource_lock'],
            output_directory=config['proof_root']+'/'+identity+'/point',
            state_root=config['child_state_root']+'/'+identity+'/point',source_receipts=copy.deepcopy(pins),
            boundary='Full original source-runtime PSM model, complete3epochs/1554minibatches and all saved raw-aligned scores/15controls. Original historical training has no persisted resource receipt and its guard history remains unproven. This full point audit is separately guarded. No new training seed, paper equivalence or strict TAB claim.')
        p=config['configuration_root']+'/'+identity+'_point.json';exclusive_json(ROOT/p,point_config)
        native_config=copy.deepcopy(native)
        native_config.update(recovery_queue=binding_path,recovery_job_id=identity,
            independent_completion=point_config['output_directory']+'/independent_full_recovery_completion.json',
            state_root=config['child_state_root']+'/'+identity+'/native',
            artifact_directory=config['artifact_root']+'/'+identity+'/native',
            rng_probe_output=point_config['output_directory']+'/original_sampler_rng_probe.json',
            source_receipts=copy.deepcopy(pins),
            boundary='Full canonical original source-runtime PSM model native checkpoint re-inference of all overlapping training and nonoverlapping test scores using original batch256,width512,CPU2 and independently reconstructed exact DFA/sampler RNG. Original framework,32GiBcommit/24GiBphysical admission,16GiBemergency and all tolerance limits retained. Historical training guard receipt missing; no journal/TAB claim and no new training seed.')
        n=config['configuration_root']+'/'+identity+'_native_prototype.json';exclusive_json(ROOT/n,native_config)
        config['cases'].append(dict(canonical_id=identity,point_configuration=p,native_configuration=n))
        sources.update((p,n))
    config['source_receipts']=[dict(path=p,sha256=digest(p)) for p in sorted(sources)]
    verify(config)
    path='configs/runtime/fm_pi_source1107_completed_full_replay.v1.json';exclusive_json(ROOT/path,config)
    root=ROOT/'projects/flow_matching_research/results/2026-10-15/pi_source1107_completed_full_replay_v1'
    exclusive_json(root/'registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=path,configuration_sha256=digest(path),original_scope=config['original_scope'],
        required_original_jobs=1140,original_required_training_seeds=[42,1103,1104,1105,1106,1107],
        ready_already_complete_models=IDS,original_full_epochs_per_model=3,original_optimizer_minibatches_per_model=1554,
        original_parameter_count_per_model=1930289,original_historical_training_resource_receipt_available=False,
        required_new_guarded_full_point_native_and_independent_review=True,
        completed_full_native_model_reinference=False,additional_training_seed_slots=0,
        new_numeric_results_published=0,all_original_experiments_complete=False,source_pins=len(config['source_receipts'])))
    print(json.dumps(dict(configuration=path,required_original_jobs=1140,ready_original_models=IDS,source_pins=len(config['source_receipts']))))


if __name__=='__main__':main()
