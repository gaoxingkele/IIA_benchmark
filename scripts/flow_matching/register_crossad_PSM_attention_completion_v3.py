"""Correct the admission test's tolerance type; preserve all native recipes."""
from copy import deepcopy
from datetime import datetime,timezone
from scripts.flow_matching import run_crossad_completion_replay_queue_v1 as full
from scripts.flow_matching import register_crossad_PSM_attention_completion_v2 as previous

NATIVE='configs/reproducibility/fm_crossad_PSM_attention_lifetime_native_replay.2026-10-12_v6.json'
POST='configs/reproducibility/fm_crossad_PSM_attention_lifetime_independent_review.2026-10-12_v6.json'
CONFIG='configs/runtime/fm_crossad_PSM_attention_lifetime_completion.v3.json'
PREVIOUS,DIAGNOSTIC=previous.PREVIOUS,previous.DIAGNOSTIC
RESULT='projects/flow_matching_research/results/2026-10-12/crossad_PSM_attention_lifetime_replay_v6'


def register():
    from scripts.flow_matching.run_crossad_PSM_lifetime_completion_v1 import verify
    failed_state='experiments/runs/fm_crossad_PSM_attention_lifetime_dispatch_v2/status.json'
    failed=full.read(failed_state)
    if failed['state']!='failure_preserved' or failed['exception']!='Preserve the failed full-source registration validation':
        raise ValueError('Require the actual failed registration, not an inferred native failure')
    old_config=full.read(previous.CONFIG);old_native=full.read(previous.NATIVE)
    if (full.ROOT/old_native['state_root']/'receipt.json').exists() or (full.ROOT/old_native['artifact_directory']).exists():
        raise ValueError('The prior admission failed before any scientific native replay; never replace an actual attempt')
    diag=full.read(old_config['diagnostic_terminal'])
    if diag['exit_code']!=0 or not diag['passed'] or diag['resource_guard_aborted'] or diag['diagnostic_checks']!=16:
        raise ValueError('Keep the passed sixteen numerical diagnostics; do not rerun until pass')
    for path in (NATIVE,POST,CONFIG):
        if (full.ROOT/path).exists():raise FileExistsError('Preserve every versioned registration')
    sources={p['path'] for p in old_native['source_receipts']}|{previous.CONFIG,previous.NATIVE,previous.POST,failed_state,
        'experiments/runs/fm_crossad_PSM_attention_lifetime_dispatch_v2/registration.stdout.log',
        previous.RESULT+'/actual_full_registration_validation.xml',
        'scripts/flow_matching/register_crossad_PSM_attention_completion_v3.py',
        'scripts/flow_matching/run_crossad_PSM_attention_dispatch_v3.py',
        'tests/test_fm_crossad_PSM_attention_registration_v3.py'}
    pins=lambda paths:[dict(path=p,sha256=full.digest(p)) for p in sorted(paths)]
    native=deepcopy(old_native)
    native.update(artifact_directory='F:/aicoding/IIA_Data/experiments/crossad_PSM_attention_lifetime_replay_v6/native',
        state_root='experiments/runs/fm_crossad_PSM_attention_lifetime_replay_v6/native',source_receipts=pins(sources))
    full.exclusive_configuration(NATIVE,native)
    post=deepcopy(full.read(previous.POST))
    post.update(native_configuration=NATIVE,state_root='experiments/runs/fm_crossad_PSM_attention_lifetime_replay_v6/post',
        output_directory=RESULT+'/independent',source_receipts=pins(sources|{NATIVE}))
    full.exclusive_configuration(POST,post)
    config=deepcopy(old_config)
    config.update(native_configuration=NATIVE,post_configuration=POST,
        state_root='experiments/runs/fm_crossad_PSM_attention_lifetime_completion_v3',
        source_receipts=pins(sources|{NATIVE,POST,old_config['original_completion_configuration']}))
    verify(config);full.exclusive_configuration(CONFIG,config)
    full.exclusive_configuration(RESULT+'/actual_type_corrected_full_PSM_registration.json',dict(
        captured_utc=datetime.now(timezone.utc).isoformat(),configuration=CONFIG,
        original_scope=full.SCOPE,required_original_jobs=139,
        preserved_failed_admission=failed,prior_scientific_native_replay_not_started=True,
        actual_native_attention_diagnostics_passed=16,
        same_model_batch_data_all_guards_original_scores_and_metrics_retained=True,
        frozen_metric_tolerance=config['metric_tolerance'],frozen_score_tolerance=config['score_tolerance'],
        test_type_correction_only=True,no_native_algorithm_or_tolerance_change=True,
        no_scientific_worker_stopped_or_modified=True,full_native_PSM_replay_passed=False,
        all_original_experiments_complete=False))
    return config


if __name__=='__main__':print(register()['state_root'])
