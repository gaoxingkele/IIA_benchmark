"""Preserve failed full PSM attempts and register the new last-use policy."""
from copy import deepcopy
from datetime import datetime,timezone
from scripts.flow_matching import run_crossad_completion_replay_queue_v1 as full
from scripts.flow_matching.register_crossad_PSM_lifetime_completion_v1 import CONFIG as PREVIOUS

NATIVE='configs/reproducibility/fm_crossad_PSM_attention_lifetime_native_replay.2026-10-12_v5.json'
POST='configs/reproducibility/fm_crossad_PSM_attention_lifetime_independent_review.2026-10-12_v5.json'
CONFIG='configs/runtime/fm_crossad_PSM_attention_lifetime_completion.v2.json'
DIAGNOSTIC='configs/reproducibility/fm_crossad_attention_lifetime_native_diagnostics.2026-10-12_v2.json'
RESULT='projects/flow_matching_research/results/2026-10-12/crossad_PSM_attention_lifetime_replay_v5'


def register():
    from scripts.flow_matching.run_crossad_PSM_lifetime_completion_v1 import verify
    previous=full.read(PREVIOUS);old=full.read(previous['native_configuration'])
    failed_path=old['state_root']+'/receipt.json';failed=full.read(failed_path)
    if failed['exit_code']==0 or failed['full_native_model_replay_passed'] or not failed['resource_guard_aborted']:
        raise ValueError('Require the actual previous full native replay failure')
    if failed['peak_tree_rss_bytes']<=old['settings']['maximum_worker_rss_bytes']:
        raise ValueError('This new final-use policy addresses the observed original RSS guard')
    diagnostic=full.read(DIAGNOSTIC);terminal_path=diagnostic['state_root']+'/receipt.json'
    terminal=full.read(terminal_path);owner=full.read(diagnostic['state_root']+'/status.json')
    if (terminal['exit_code']!=0 or not terminal['passed'] or terminal['resource_guard_aborted']
            or terminal['diagnostic_checks']!=16 or terminal['original_replay_settings']!=old['settings']
            or terminal['configuration_sha256']!=full.digest(DIAGNOSTIC)):
        raise ValueError('Exact source/mask/gradient/dropout/final-use diagnostics under original guards are required')
    for path in (NATIVE,POST,CONFIG):
        if (full.ROOT/path).exists():raise FileExistsError('Keep every registered prior attempt immutable')
    sources={p['path'] for p in old['source_receipts']}|{PREVIOUS,previous['native_configuration'],failed_path,
        DIAGNOSTIC,terminal_path,diagnostic['output_directory']+'/native_attention_lifetime_diagnostic_validation.xml',
        'src/iia_benchmark/models/fm_crossad_attention_lifetime_v2.py',
        'scripts/flow_matching/replay_crossad_attention_lifetime_v5.py',
        'scripts/flow_matching/run_crossad_PSM_attention_completion_v2.py',
        'scripts/flow_matching/run_crossad_attention_lifetime_diagnostics_v2.py',
        'scripts/flow_matching/register_crossad_PSM_attention_completion_v2.py',
        'tests/test_fm_crossad_attention_lifetime_source_v2.py',
        'tests/test_fm_crossad_attention_lifetime_native_v2.py',
        'tests/test_fm_crossad_PSM_attention_registration_v2.py'}
    pins=lambda paths:[dict(path=p,sha256=full.digest(p)) for p in sorted(paths)]
    native=deepcopy(old)
    native.update(artifact_directory='F:/aicoding/IIA_Data/experiments/crossad_PSM_attention_lifetime_replay_v5/native',
        state_root='experiments/runs/fm_crossad_PSM_attention_lifetime_replay_v5/native',source_receipts=pins(sources))
    native['boundary']+=' Additionally release only attn_scores after the final original F.softmax use. No operation, mask, dropout draw, checkpoint, batch, original score tolerance1e-6/1e-7 or metric tolerance1e-12 changes. Actual16 guarded numerical diagnostics passed; full native and independent completion remain unproved.'
    full.exclusive_configuration(NATIVE,native)
    post=deepcopy(full.read(previous['post_configuration']))
    post.update(native_configuration=NATIVE,state_root='experiments/runs/fm_crossad_PSM_attention_lifetime_replay_v5/post',
        output_directory=RESULT+'/independent',source_receipts=pins(sources|{NATIVE}))
    full.exclusive_configuration(POST,post)
    config=deepcopy(previous)
    config.update(native_configuration=NATIVE,post_configuration=POST,
        state_root='experiments/runs/fm_crossad_PSM_attention_lifetime_completion_v2',
        diagnostic_terminal=terminal_path,diagnostic_owner={k:owner[k] for k in ('pid','create_time','command')},
        source_receipts=pins(sources|{NATIVE,POST,previous['original_completion_configuration']}))
    verify(config);full.exclusive_configuration(CONFIG,config)
    full.exclusive_configuration(RESULT+'/actual_full_PSM_attention_lifetime_registration.json',dict(
        captured_utc=datetime.now(timezone.utc).isoformat(),configuration=CONFIG,original_scope=full.SCOPE,
        required_original_jobs=139,unchanged_other_original_job_recipes=138,
        full_parameters=1328459,original_batch=128,original_seq_len=192,
        full_original_points=dict(train=105984,validation=26497,test=87841),
        previous_actual_failed_native_replay=failed,
        actual_guarded_source_numerical_diagnostics=terminal,
        diagnostics_are_not_full_PSM_performance=True,
        raw_point_certificate_reused=True,all_original_guards_score_metric_tolerances_retained=True,
        every_source_arithmetic_operation_and_original_batch_retained=True,
        additional_training_seed_slots=0,no_actual_scientific_child_modified_or_stopped=True,
        full_native_PSM_replay_passed=False,all_original_experiments_complete=False))
    return config


if __name__=='__main__':print(register()['state_root'])
