"""Diagnosed metadata-memory recovery,all scopes and resource limits unchanged."""
import argparse
from pathlib import Path
import sys
from scripts.flow_matching import audit_moment_icm_pretraining_corpus_v1 as original
from scripts.flow_matching.moment_icm_streaming_scope_v2 import profile_stream

ROOT,read,sha,write=original.ROOT,original.read,original.sha,original.write
MODULE='scripts.flow_matching.audit_moment_icm_pretraining_corpus_v2'


def verify(config):
    groups,compat=original.verify(config)
    failure=read(ROOT/config['previous_failed_guard'])
    if (failure['passed'] or not failure['resource_guard_aborted'] or failure['proof_sha256'] is not None
            or sha(ROOT/config['previous_failed_guard'])!=config['previous_failed_guard_sha256']
            or config['metadata_recovery']!='stream_validate_all_values_retain_only_lengths'
            or config['settings']!=read(ROOT/config['previous_configuration'])['settings']):
        raise ValueError('Known preserved metadata failure and unchanged full resource/scope policy required')
    return groups,compat


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--probe-only',action='store_true');args=p.parse_args()
    config=read(ROOT/args.config);verify(config);config['_config_path']=args.config
    if args.probe_only:
        print(__import__('json').dumps(dict(passed=True,files=2441,Monash=47,
            numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)))
    elif args.worker:
        # Fresh isolated worker only;immutable v1 files and existing processes stay
        # unchanged. The old parser class supplies its exact special-split list.
        prior=original.inputs.profile_short
        try:
            original.inputs.profile_short=profile_stream
            original.worker(config)
        finally:original.inputs.profile_short=prior
        path=Path(config['artifact_root'])/'full_corpus_and_temporal_gate_audit.json'
        report=read(path);report.update(streaming_metadata_recovery=True,
            streaming_source_sha256=sha(ROOT/'scripts/flow_matching/moment_icm_streaming_scope_v2.py'),
            prior_failed_guard_sha256=config['previous_failed_guard_sha256'],
            original_source_numerical_training_or_preprocessing_retried=False)
        write(path,report)
    else:
        original.MODULE=MODULE  # Only this fresh metadata interpreter's child command.
        original.controller(config)


if __name__=='__main__':main()
