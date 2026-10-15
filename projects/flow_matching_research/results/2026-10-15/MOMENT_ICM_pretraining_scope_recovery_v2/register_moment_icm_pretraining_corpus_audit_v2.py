"""Freeze diagnosed streaming metadata recovery;preserve all prior evidence."""
from copy import deepcopy
import json
from scripts.flow_matching import audit_moment_icm_pretraining_corpus_v2 as s
from scripts.flow_matching.register_moment_icm_pretraining_corpus_audit_v1 import CONFIG as PRIOR

CONFIG='configs/reproducibility/fm_moment_ICM_pretraining_corpus_audit.2026-10-15_v2.json'
EVIDENCE='projects/flow_matching_research/results/2026-10-15/MOMENT_ICM_pretraining_scope_recovery_v2'


def registration():
    previous=s.read(s.ROOT/PRIOR);config=deepcopy(previous)
    guard=EVIDENCE+'/preserved_v1_original_guard.json'
    config.update(previous_configuration=PRIOR,previous_failed_guard=guard,
        previous_failed_guard_sha256=s.sha(s.ROOT/guard),
        metadata_recovery='stream_validate_all_values_retain_only_lengths',
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_pretraining_input_scope_v2',
        state_root='experiments/runs/fm_moment_icm_pretraining_input_scope_v2')
    pins={r['path'] for r in config['source_receipts']}
    pins.update([PRIOR,guard,'scripts/flow_matching/moment_icm_streaming_scope_v2.py',
        'scripts/flow_matching/audit_moment_icm_pretraining_corpus_v2.py',
        'scripts/flow_matching/register_moment_icm_pretraining_corpus_audit_v2.py',
        'tests/test_fm_moment_icm_streaming_scope_v2.py'])
    config['source_receipts']=[dict(path=p,sha256=s.sha(s.ROOT/p)) for p in sorted(pins)]
    config['boundary']+=(' Metadata-v1 audit hit its6GiBworker cap while building a large source value frame;'
        'preserved failed guard has no positive final proof. V2 changes only metadata implementation:validateevery '
        'numeric token then discard values,retain all lengths,and use the exact source pandas/NumPy seed13 row '
        'sampling on length-only data. Original special single-series splitter and its tail omissions retained. '
        'All2441files/all47profiles/context256/ratios/seeds/source-versus-temporal controls/resources unchanged. '
        'No model fit/inference or source numerical preprocessing retried. Small real-source equivalence tests '
        'do not certify the historical ICM corpus or actual pretraining. Both versions stay reviewable.')
    return config


def main():
    path=s.ROOT/CONFIG
    if path.exists():raise FileExistsError('Preserve frozen streaming recovery registration')
    config=registration();s.verify(config)
    with path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(config,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(configuration=CONFIG,files=2441,Monash=47,original_failure_preserved=True,model_runs=0)))


if __name__=='__main__':main()
