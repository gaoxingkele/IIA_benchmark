"""Metadata-corrected full source CNF;no numerical recipe change from v1."""
import copy
from types import FunctionType
from scripts.flow_matching import run_source_cnf_full120_v1 as previous

# Freeze and reuse every actual implementation function;rebind dispatch/output
# metadata only. No running/frozen v1 file or registered configuration is edited.
for _key,_value in vars(previous).items():
    if not _key.startswith('__'):
        globals()[_key]=_value
MODULE='scripts.flow_matching.run_source_cnf_full120_v2'
QUEUE='configs/experiments/fm_source_cnf_full120.v2.json'
MODEL='configs/models/fm_source_cnf_full_training.v2.json'
TEST='tests/test_fm_source_cnf_full120_v2.py'
for _key,_value in list(vars(previous).items()):
    if isinstance(_value,FunctionType) and _value.__module__==previous.__name__:
        globals()[_key]=FunctionType(_value.__code__,globals(),_key,_value.__defaults__,_value.__closure__)


def scientific_cases():
    jobs=previous.scientific_cases()
    for job in jobs:
        job['model_config']=MODEL
        job['artifact_directory']=job['artifact_directory'].replace('/source_cnf_full120_v1/','/source_cnf_full120_v2/')
        job['scientific_protocol']['validation_loss_includes_leftout_timepoint']=False
    return jobs


def registration():
    q=FunctionType(previous.registration.__code__,globals())()
    q.update(state_root='experiments/runs/fm_source_cnf_full120_v2',
        lifecycle_artifact_root='F:/aicoding/IIA_Data/experiments/source_cnf_full120_v2_framework_lifecycle',
        metadata_predecessor_configuration=previous.QUEUE,
        validation_likelihood_excludes_requested_leftout=True)
    old='Whitening/source min-size loader and validation including leftout remain as release code.'
    assert q['boundary'].count(old)==1
    q['boundary']=q['boundary'].replace(old,
        'Whitening fits all selected observations and source min-size loaders remain. Corrected CNF likelihood excludes the requested leftout index in both training and validation;full evaluation populations remain included. The v1 CFM-derived inclusion flag was incorrect metadata;no numerical equation changed.')
    paths={x['path'] for x in q['source_receipts']}|{previous.QUEUE,
        previous.MODULE.replace('.','/')+'.py',previous.MODEL,previous.TEST,
        'scripts/flow_matching/audit_source_cnf_full120_v2.py'}
    q['source_receipts']=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(paths)]
    return q


if __name__=='__main__':main()
