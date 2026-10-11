"""Freeze all original GRASP completion routes and already verified native evidence."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.flow_matching.run_grasp_completion_replay_queue_v1 import ROOT, read, sha, verify, GUARDS
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json


def main():
    template_path = 'configs/reproducibility/fm_grasp_tau11104_checkpoint_replay.2026-10-12_v2.json'
    template = read(ROOT/template_path)
    runtime = {key:copy.deepcopy(template[key]) for key in (*GUARDS, 'original_scope')}
    runtime.update(original_queue=template['queue'], native_replay_template=template_path,
        required_original_models=9120, required_original_cohorts=480, required_original_training_seeds=10,
        state_root='experiments/runs/fm_grasp_completion_replay_queue_v1',
        child_state_root='experiments/runs/fm_grasp_completion_replay_children_v1',
        child_configuration_root='configs/reproducibility/fm_grasp_completion_replay_children_v1',
        proof_root='projects/flow_matching_research/results/2026-10-12/grasp_full9120_completion_replay_v1/completed_models',
        artifact_root='F:/aicoding/IIA_Data/experiments/grasp_full9120_completion_replay_v1', poll_seconds=10,
        recovery_routes=[], preserved_verified_models=[])
    sources = {item['path'] for item in template['source_receipts']}
    sources.update([template_path, template['queue'], 'scripts/flow_matching/register_grasp_completion_replay_queue_v1.py',
        'scripts/flow_matching/run_grasp_completion_replay_queue_v1.py',
        'scripts/flow_matching/run_grasp_registered_replay_review_pipeline_v2.py',
        'tests/test_fm_grasp_completion_replay_queue_v1.py',
        'tests/test_fm_grasp_registered_replay_review_pipeline_v2.py'])
    for queue_path in ('configs/experiments/fm_native_grasp_exact_recovery.v2.json',
                       'configs/experiments/fm_native_grasp_tau025_exact_recovery.v3.json'):
        sources.add(queue_path)
        for job in read(ROOT/queue_path)['jobs']:
            runtime['recovery_routes'].append(dict(canonical_id=job['recovery_original_job_id'],
                execution_queue=queue_path, execution_id=job['id']))
    assert len(runtime['recovery_routes']) == 3
    legacy_config = 'configs/reproducibility/fm_grasp_complete_full_replay_union.2026-10-11_v1.json'
    legacy = read(ROOT/legacy_config)
    sources.add(legacy_config)
    sources.update(p['path'] for p in legacy['source_receipts'])
    old_results = 'configs/reproducibility/fm_grasp_full_checkpoint_replay.2026-10-11_v1.json'
    sources.add(old_results)
    for binding in read(ROOT/old_results)['full_results']:
        runtime['preserved_verified_models'].append(dict(binding, kind='legacy_union',
            native_configuration=legacy['replay_config'], independent_proof=legacy['output']))
    for tag in ('lin1104', 'mean1104', 'tau01104', 'tau0251103', 'tau0251104', 'tau051104'):
        proof_path = f'projects/flow_matching_research/results/2026-10-11/grasp_{tag}_full_replay_verified_v1/independent_full_replay_verification.json'
        proof = read(ROOT/proof_path)
        native_path = proof['completed_native_replay_configuration']
        item = read(ROOT/native_path)['full_results'][0]
        runtime['preserved_verified_models'].append(dict(item, kind='lin' if tag == 'lin1104' else 'generic',
            native_configuration=native_path, independent_proof=proof_path))
    native_path = 'configs/reproducibility/fm_grasp_tau11104_checkpoint_replay.2026-10-12_v2.json'
    runtime['preserved_verified_models'].append(dict(read(ROOT/native_path)['full_results'][0], kind='generic',
        native_configuration=native_path, independent_proof='projects/flow_matching_research/results/2026-10-12/grasp_tau11104_full_replay_review_v1/independent/independent_full_replay_verification.json'))
    for kind, native_path, proof_path in [
        ('er', 'configs/reproducibility/fm_grasp_er1104_checkpoint_replay.2026-10-11_v1.json',
         'projects/flow_matching_research/results/2026-10-11/cfm_node1104_and_grasp_er1104_native_verification_v1/independently_verified_increment.json'),
        ('transformer', 'configs/reproducibility/fm_grasp_transformer_seed1104_full_checkpoint_replay.2026-10-11_v1.json',
         'projects/flow_matching_research/results/2026-10-11/grasp_transformer_seed1104_full_checkpoint_replay_v1/independent_full_array_validation.json')]:
        runtime['preserved_verified_models'].append(dict(read(ROOT/native_path)['full_results'][0], kind=kind,
            native_configuration=native_path, independent_proof=proof_path))
    assert len(runtime['preserved_verified_models']) == 22
    for item in runtime['preserved_verified_models']:
        sources.update([item['result_path'], item['native_configuration'], item['independent_proof']])
        producer = read(ROOT/item['native_configuration'])
        sources.update(p['path'] for p in producer['source_receipts'])
        sources.update([producer['state_root']+'/receipt.json', producer['output_directory']+'/checkpoint_replay_audit.json'])
    runtime['source_receipts'] = [dict(path=p, sha256=sha(ROOT/p)) for p in sorted(sources)]
    runtime['boundary'] = ('All9120 original GRASP entity jobs/480cohorts/10seeds preserved. Three completed exact recovery '
        'executions replace only their execution paths. Preserve22 completed original native+independent proofs. '
        'Remaining original results enter unchanged full-checkpoint reconstruction, every original complete score recipe '
        'and independent full-array/point-identity/forward-count/metric review automatically when original full training '
        'really completes. No retraining, added seeds, shortened budgets, changed tolerances/resource gates, automatic '
        'numeric publication, author certification or strictTAB claim. Preserve all failed attempts and raw artifacts.')
    originals, executions, _, covered = verify(runtime)
    ready = []
    for canonical, job in executions.items():
        if canonical in covered:
            continue
        result_path = ROOT/job['output_directory']/'result.json'
        if result_path.exists() and read(result_path).get('status') == 'completed':
            ready.append(canonical)
    path = 'configs/runtime/fm_grasp_completion_replay_queue.v1.json'
    exclusive_json(ROOT/path, runtime)
    evidence = ROOT/'projects/flow_matching_research/results/2026-10-12/grasp_full9120_completion_replay_v1'
    exclusive_json(evidence/'registration.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=path, configuration_sha256=sha(ROOT/path), required_original_models=len(originals),
        original_cohorts=480, required_original_training_seeds=10,
        previously_complete_native_and_independent_models=len(covered),
        remaining_original_models=len(originals)-len(covered), currently_complete_but_unreviewed_original_models=ready,
        preserved_recovery_routes=runtime['recovery_routes'], source_pins=len(sources),
        original_scope=runtime['original_scope'], all_original_experiments_complete=False,
        author_equivalence_certified=False, strict_TAB_result=False, numeric_publication_unchanged=True))
    print(json.dumps(dict(configuration=path, registered_original_models=len(originals),
        preserved_completed_models=len(covered), remaining_original_models=len(originals)-len(covered),
        complete_but_unreviewed_models=ready, source_pins=len(sources), all_original_experiments_complete=False)))


if __name__ == '__main__':
    main()
