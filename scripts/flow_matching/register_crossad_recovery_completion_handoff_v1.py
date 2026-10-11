"""Bind all four exact recovery outputs to the immutable139 original jobs."""
import copy
from datetime import datetime,timezone
import json

from scripts.flow_matching.run_crossad_recovery_completion_handoff_v1 import ROOT,read,digest,verify
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json


def main():
    base_path='configs/runtime/fm_crossad_completion_replay_queue.v1.json'
    base=read(base_path)
    config={k:copy.deepcopy(base[k]) for k in ('python','cpu_threads','settings','resource_lock',
        'original_scope','required_original_jobs','score_tolerance','metric_tolerance','poll_seconds')}
    config.update(original_completion_configuration=base_path,
        state_root='experiments/runs/fm_crossad_recovery_completion_handoff_v1',
        child_state_root='experiments/runs/fm_crossad_recovery_completion_children_v1',
        configuration_root='configs/reproducibility/fm_crossad_recovery_completion_children_v1',
        proof_root='projects/flow_matching_research/results/2026-10-12/crossad_exact_recovery_completion_handoff_v1/completed_models',
        artifact_root='F:/aicoding/IIA_Data/experiments/crossad_exact_recovery_completion_handoff_v1',recovery_routes=[])
    sources={p['path'] for p in base['source_receipts']}|{base_path,
        'scripts/flow_matching/run_crossad_recovery_completion_handoff_v1.py',
        'scripts/flow_matching/register_crossad_recovery_completion_handoff_v1.py',
        'tests/test_fm_crossad_recovery_completion_handoff_v1.py'}
    for path in ('configs/experiments/fm_crossad_resource_recovery.v1.json',
                 'configs/experiments/fm_crossad_resource_recovery.v2.json'):
        registration=read(path)
        sources.update(p['path'] for p in registration['source_receipts'])
        sources.update((path,registration['native_queue']))
        for case in registration['jobs']:
            config['recovery_routes'].append(dict(canonical_id=case['original_id'],recovery_job_id=case['job']['id'],
                native_queue=registration['native_queue'],recovery_registration=path,kind='registered_exact_recovery',
                original_failure_path=case['original_failure_path'],original_failure_sha256=case['original_failure_sha256']))
            sources.add(case['original_failure_path'])
    path='configs/runtime/fm_crossad_shared_commit_full_retry.v1.json';registration=read(path)
    sources.update(p['path'] for p in registration['source_receipts'])
    sources.update((path,registration['queue'],registration['terminal_incident_evidence'],
        'scripts/flow_matching/run_selected_native_full_retry_v1.py'))
    config['recovery_routes'].append(dict(canonical_id=registration['selected_job_id'],
        recovery_job_id=registration['selected_job_id'],native_queue=registration['queue'],recovery_registration=path,
        kind='selected_full_retry',selected_state_root=registration['state_root'],
        original_failure_path=registration['terminal_incident_evidence'],
        original_failure_sha256=digest(registration['terminal_incident_evidence'])))
    config['source_receipts']=[dict(path=p,sha256=digest(p)) for p in sorted(sources)]
    _,jobs,_,covered,routes=verify(config)
    path='configs/runtime/fm_crossad_recovery_completion_handoff.v1.json'
    exclusive_json(ROOT/path,config)
    evidence=ROOT/'projects/flow_matching_research/results/2026-10-12/crossad_exact_recovery_completion_handoff_v1'
    exclusive_json(evidence/'registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=path,configuration_sha256=digest(path),original_scope=config['original_scope'],
        required_original_jobs=len(jobs),preserved_verified_original_jobs=len(covered),pending_original_jobs=len(jobs)-len(covered),
        exact_recovery_routes=config['recovery_routes'],source_pins=len(sources),
        original_completion_controller_unchanged=True,original_training_controllers_unchanged=True,
        required_full_checkpoint_reinference_and_full_independent_review=True,
        original_model_data_training_evaluator_settings_and_all_budgets_unchanged=True,
        numeric_publication_unchanged=True,additional_training_seed_slots=0,all_original_experiments_complete=False))
    print(json.dumps(dict(configuration=path,required_original_jobs=len(jobs),exact_recovery_routes=len(routes),
        preserved_verified_original_jobs=len(covered),pending_original_jobs=len(jobs)-len(covered),source_pins=len(sources))))


if __name__=='__main__':main()
