"""Register all saved original stages for additional full native re-inference."""
import copy
from datetime import datetime, timezone
import json

from scripts.flow_matching.audit_timegan_saved_predecessor_native_v1 import original, verify, compiled_worker


def main():
    path = 'configs/runtime/fm_timegan_saved_predecessor_native_audit.v1.json'
    recovery_path = 'configs/reproducibility/fm_timegan_stock1104_posthoc_recovery.2026-10-12_v1.json'
    recovery = original.read(original.ROOT / recovery_path)
    reference_path = 'configs/runtime/fm_timegan_energy_stage_restore_audit.v1.json'
    reference = original.read(original.ROOT / reference_path)
    sources = {p['path'] for p in reference['source_receipts']} | {p['path'] for p in recovery['source_receipts']}
    sources.update([recovery_path, reference_path,
        'scripts/flow_matching/audit_timegan_saved_predecessor_native_v1.py',
        'scripts/flow_matching/run_timegan_saved_predecessor_native_audit_v1.py',
        'scripts/flow_matching/register_timegan_saved_predecessor_native_audit_v1.py',
        'tests/test_fm_timegan_saved_predecessor_native_v1.py'])
    config = dict(recovery_configuration=recovery_path, reference_audit_runtime=reference_path,
        canonical_job=recovery['canonical_job'], original_scope=recovery['original_scope'],
        settings=copy.deepcopy(reference['settings']), resource_lock=reference['resource_lock'], python=reference['python'],
        state_root='experiments/runs/fm_timegan_saved_predecessor_native_audit_v1',
        output_root='projects/flow_matching_research/results/2026-10-12/timegan_saved_predecessor_native_audit_v1/native',
        source_receipts=[dict(path=p,sha256=original.sha(original.ROOT/p)) for p in sorted(sources)],
        citation='Yoon, Jarrett, van der Schaar. Time-series Generative Adversarial Networks. NeurIPS 2019.',
        code='https://github.com/jsyoon0823/TimeGAN',
        boundary='Additional complete re-inference of all actual13 saved successful native checkpoints and ten '
            'trained discriminators. No trained predictor exists yet, none is substituted or fabricated. All full '
            'predictor training and original full45/all20 final native replay obligations remain unchanged. '
            'Existing read-only audit safeguards retained, original source/data/checkpoints preserved, zero new '
            'optimizer updates or training seed slots. This audit is not full experiment completion or benchmark publication.')
    verify(config); compiled_worker()
    original.exclusive(original.ROOT / path, config)
    output = original.ROOT / 'projects/flow_matching_research/results/2026-10-12/timegan_saved_predecessor_native_audit_v1'
    original.exclusive(output/'registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=path,configuration_sha256=original.sha(original.ROOT/path),original_scope=config['original_scope'],
        all_actual_saved_native_checkpoints_required=13,complete_generator_windows_required=3661,
        actual_saved_trained_discriminator_replays_required=10,all_saved_source_protocol_iterations_required=20000,
        bitwise_gate=True,additional_training_seed_slots=0,training_executed=False,original_audit_guards_unchanged=True,
        final_full45_all20_replay_obligations_unchanged=True,experiment_complete=False,new_benchmark_results=0))
    print(json.dumps(dict(configuration=path,saved_native_checkpoints=13,full_original_jobs=45)))


if __name__ == '__main__':
    main()
