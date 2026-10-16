"""Isolated full139 continuation after the actual unused-PATE import failure.

Keep every original model/data/budget. Reuse the real completed20 recovery for
GECCO1105, and select only the already verified optional-metric evaluator patch.
"""
import argparse
import copy
from types import FunctionType
from scripts.flow_matching import run_crossad_original_CUDA_full139_v1 as parent
from scripts.flow_matching import recover_crossad_original_CUDA_completed20_v1 as recovery
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive

MODULE='scripts.flow_matching.run_crossad_original_CUDA_full139_v2'
CONFIG='configs/experiments/fm_crossad_original_CUDA_full139.v2.json'


def registration():
    previous=read(ROOT/parent.CONFIG);c=copy.deepcopy(previous);r=read(ROOT/recovery.CONFIG)
    c.update(schema_version=2,state_root='experiments/runs/fm_crossad_original_CUDA_full139_v2',
        artifact_root='F:/aicoding/IIA_Data/experiments/crossad_original_CUDA_full139_v2',
        delegated_first_configuration=recovery.CONFIG,optional_evaluator_root=r['optional_evaluator_root'],
        preserved_parent_configuration=parent.CONFIG,preserved_parent_configuration_sha256=sha(ROOT/parent.CONFIG),
        original_failed_full139_attempt_preserved=True,completed20_training_reused_without_additional_updates=True)
    c['settings']['output_root']=c['state_root']
    for job in c['jobs']:
        canonical=job['canonical_CPU_condition_reference']
        job.update(id=canonical+'__original_CUDA_FP32_full139_v2',runner=MODULE,
            output_directory=c['state_root']+'/jobs/'+canonical+'/author_job')
    pins={v['path']:v['sha256'] for v in previous['source_receipts']}
    for path in [parent.CONFIG,recovery.CONFIG,MODULE.replace('.','/')+'.py',
                 recovery.MODULE.replace('.','/')+'.py','tests/test_fm_crossad_original_CUDA_full139_v2.py']:
        pins[path]=sha(ROOT/path)
    c['source_receipts']=[dict(path=p,sha256=h) for p,h in sorted(pins.items())]
    return c


def verify(c):
    if c!=registration():raise ValueError('Full139 isolated continuation must preserve every original condition and gate')
    for item in c['source_receipts']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Frozen full139 continuation dependency changed')
    if len(c['jobs'])!=139 or len({j['canonical_CPU_condition_reference'] for j in c['jobs']})!=139:
        raise ValueError('All139 source conditions required')


def controller_callable():
    return FunctionType(parent.controller.__code__,dict(parent.controller.__globals__,verify=verify,MODULE=MODULE),
        argdefs=parent.controller.__defaults__)


def dispatch(c,job,phase):
    if phase in ['CUDA_probe','CPU_reference']:
        parent.native_probe(c,job,cpu=phase=='CPU_reference')
    else:
        recovery.install_evaluator(c)
        parent.source_or_replay(c,job,replay=phase=='replay')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true')
    p.add_argument('--phase',choices=parent.PHASES);p.add_argument('--canonical-id');a=p.parse_args()
    if a.register:exclusive(ROOT/CONFIG,registration());return
    c=read(ROOT/CONFIG);verify(c)
    if a.phase:dispatch(c,next(j for j in c['jobs'] if j['canonical_CPU_condition_reference']==a.canonical_id),a.phase)
    else:controller_callable()(c)


if __name__=='__main__':main()
