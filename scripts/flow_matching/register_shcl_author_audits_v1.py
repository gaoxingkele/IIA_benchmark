"""Bind acquired native Torch1.11 to unchanged full13 operator/full52 data audits."""
import json
from pathlib import Path
import subprocess

from scripts.flow_matching import shcl_released_native_v1 as native
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    acquisition='configs/acquisition/fm_shcl_author_environment.v1.json'; environment=native.read(native.ROOT/acquisition)
    receipt=environment['root']+'/environment_acquisition_report.json'
    report=native.read(receipt)
    if not report['offline_install_complete'] or not report['pip_check_passed'] or not report['all_publisher_SHA256_verified']:
        raise ValueError('Complete actual native packages required')
    actual=json.loads(subprocess.check_output([environment['python'],'-X','utf8','-m',
        'scripts.flow_matching.shcl_author_audits_v1','--environment-only'],cwd=native.ROOT,encoding='utf-8'))
    for kind,predecessor,path in [('operators','configs/runtime/fm_shcl_released_operator_audit.v1.json',
        'configs/runtime/fm_shcl_author_operators.v1.json'),('data','configs/datasets/fm_shcl_released_data_views.v3.json',
        'configs/runtime/fm_shcl_author_data.v1.json')]:
        config=native.read(native.ROOT/predecessor)
        extra=[predecessor,acquisition,receipt,'scripts/flow_matching/shcl_author_audits_v1.py',
            'scripts/flow_matching/run_shcl_author_audits_v1.py','scripts/flow_matching/register_shcl_author_audits_v1.py']
        config['source_receipts'].extend(dict(path=p,sha256=native.sha(native.ROOT/p)) for p in extra)
        config.update(audit_kind=kind,environment=actual,python=environment['python'],
            state_root='experiments/runs/fm_shcl_author_'+kind+'_v1',
            original_native_environment_equivalence=False,
            environment_boundary='Declared Torch1.11 and every imported numerical dependency except explicit NumPy1.26.4 repair. '
                'Both original full audit code objects and frozen tolerances/guards unchanged; metadata bindings differ only.')
        if kind=='operators': config['output_root']='projects/flow_matching_research/results/2026-10-12/shcl_author_environment_v1/operators'
        else: config['view_root']='F:/aicoding/IIA_Data/experiments/shcl_released_native_v1/data_views_author_v1'
        exclusive(native.ROOT/path,config)
    print(json.dumps(dict(original_operator_profiles=13,original_data_loader_modes=52,actual_environment=actual)))


if __name__=='__main__':main()
