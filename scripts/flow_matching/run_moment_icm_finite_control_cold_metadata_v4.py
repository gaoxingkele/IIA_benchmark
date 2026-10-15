"""Fresh cold metadata guard for the unchanged v3 data worker.

The original v3 metadata entry failed before creating any scientific worker:
it installed resource facades twice. Keep that failure and install them exactly
once through the original guarded controller. No numerical worker is retried.
"""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import sys
from scripts.flow_matching import prepare_moment_icm_anomaly_finite_control_v3 as worker

ROOT,read,sha,write=worker.ROOT,worker.read,worker.sha,worker.write
CONFIG='configs/reproducibility/fm_moment_ICM_all322_finite_control_cold_metadata.2026-10-15_v4.json'


def require_pre_science_failure(base,artifact):
    base=Path(base);artifact=Path(artifact)
    text=(base/'controller.stderr.log').read_text(encoding='utf-8')
    if ('Install resource facades only in a fresh cold metadata interpreter' not in text
            or (base/'worker_launch.json').exists() or artifact.exists()):
        raise ValueError('Only a diagnosed pre-science metadata failure can be replaced')


def registration():
    original=read(ROOT/worker.CONFIG);worker.verify(original)
    base=ROOT/original['state_root'];require_pre_science_failure(base,original['artifact_root'])
    pins=[worker.CONFIG,'scripts/flow_matching/run_moment_icm_finite_control_cold_metadata_v4.py',
        'tests/test_fm_moment_icm_finite_control_cold_metadata_v4.py',
        str(base/'controller_launch.json'),str(base/'controller.stderr.log')]
    return dict(schema_version=4,worker_configuration=worker.CONFIG,worker_configuration_sha256=sha(ROOT/worker.CONFIG),
        required_sources=322,settings=original['settings'],python=original['python'],resource_lock=original['resource_lock'],
        artifact_root=original['artifact_root'],state_root='experiments/runs/fm_moment_icm_anomaly_finite_control_cold_metadata_v4',
        original_pre_science_metadata_failure=dict(state_root=original['state_root'],stderr_sha256=sha(base/'controller.stderr.log')),
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in pins],
        boundary='Only cold metadata API initialization repaired;unchanged v3 frozen scientific worker/config/data transforms,322cases,original resource policy. Old failed metadata handle/logs preserved;no old scientific worker existed,none retried. No historical source equivalence or neural performance claim.')


def pre_guard(runtime):
    for pin in runtime['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen cold metadata entry changed: '+pin['path'])
    original=read(ROOT/worker.CONFIG);worker.verify(original)
    if (runtime['worker_configuration']!=worker.CONFIG or runtime['worker_configuration_sha256']!=sha(ROOT/worker.CONFIG)
            or runtime['required_sources']!=322 or runtime['settings']!=original['settings']
            or runtime['resource_lock']!=original['resource_lock'] or runtime['python']!=original['python']
            or runtime['artifact_root']!=original['artifact_root']):
        raise ValueError('Exact unchanged v3 scientific worker and original guards required')
    if not worker.audit_prerequisite(original):raise ValueError('Actual full readonlyaudit prerequisite must be complete')
    return original


def controller(runtime):
    pre_guard(runtime)
    # This function is the only entry that installs resource facades.
    path=ROOT/'scripts/flow_matching/verify_moment_icm_PILE_anomaly_v1.py'
    node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='controller')
    replacements={'closing_all1949_anomaly_sources':'preparing_all322_explicit_finite_controls',
        'all1949_anomaly_cache_closure_passed':'all322_finite_control_preparation_passed'}
    class Names(ast.NodeTransformer):
        def visit_Constant(self,n):
            if isinstance(n.value,str) and n.value in replacements:return ast.copy_location(ast.Constant(replacements[n.value]),n)
            return n
    node=Names().visit(node);ast.fix_missing_locations(node)
    ns=dict(worker.guard_template.controller.__globals__)
    ns.update(MODULE=worker.MODULE,CONFIG=CONFIG,REPORT=worker.REPORT)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path)+'::<cold-v4-data-worker-guard>','exec'),ns)
    ns['controller'](runtime)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true');parser.add_argument('--probe-only',action='store_true');args=parser.parse_args()
    if args.register:
        if (ROOT/CONFIG).exists():raise FileExistsError('Preserve frozen cold metadata registration')
        runtime=registration();pre_guard(runtime)
        with (ROOT/CONFIG).open('x',encoding='utf-8',newline='\n') as stream:json.dump(runtime,stream,indent=2);stream.write('\n')
    else:runtime=read(ROOT/CONFIG);pre_guard(runtime)
    if args.probe_only:print(json.dumps(dict(passed=True,unchanged_scientific_worker=worker.MODULE,
        required_sources=322,numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)))
    else:controller(runtime)


if __name__=='__main__':main()
