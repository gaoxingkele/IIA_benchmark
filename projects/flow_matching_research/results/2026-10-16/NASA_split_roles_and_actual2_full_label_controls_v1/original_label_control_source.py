"""Separate fixed-score label experiment under a declared exact CSV transform.

Keep the original direct-float gate/results unchanged. Only G-7 uses the fixed
Pandas CSV/default-C reader representation proved by the complete96-way audit.
This does not certify the unknown historical generator or select using F1.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from scripts.flow_matching import audit_timesnet_NASA_label_protocol_v2 as previous
from scripts.flow_matching import audit_NASA_split_representation_v1 as characterization

ROOT,read,sha=previous.ROOT,previous.read,previous.sha
MODULE='scripts.flow_matching.audit_timesnet_declared_CSV_label_control_v1'
CONFIG='configs/reproducibility/fm_timesnet_all5_declared_CSV_label_control.2026-10-16_v1.json'
TEST='tests/test_fm_timesnet_declared_CSV_label_control_v1.py'
PROFILE='CSV_pandas_default_C_read'


def representation_evidence():
    c=read(ROOT/characterization.CONFIG);characterization.verify(c)
    p=Path(c['artifact_root'])/characterization.REPORT;g=ROOT/c['state_root']/'guard_receipt.json'
    if not p.exists() or not g.exists():raise ValueError('Actual complete preregistered representation audit required')
    proof,guard=read(p),read(g)
    if (not proof['passed'] or proof['configuration_sha256']!=sha(ROOT/characterization.CONFIG)
            or proof['declared_comparisons_executed']!=96 or not guard['passed'] or guard['exit_code']!=0
            or guard['resource_guard_aborted'] or guard['proof_sha256']!=sha(p)):
        raise ValueError('All96actual comparisons and successful original readonly guard required')
    entity=next(r for r in proof['results'] if r['entity']=='SMAP_G-7')
    for role in ['val','test']:
        matches=[r for r in entity['CSV_roles'][role]['candidates'] if r['every_coordinate_equal']]
        if len(matches)!=1 or matches[0]['profile']!=PROFILE or matches[0]['raw_role']!='test' or matches[0]['timestamp_origin']!=1:
            raise ValueError('Unique exact INPUT-value-only G7 transformation match required')
    partitions=entity['complete_raw_TEST_partitions']
    if len(partitions)!=1 or not partitions[0]['every_original_test_row_once'] or partitions[0]['overlap_rows']!=0:
        raise ValueError('All original G7 TEST rows must partition exactly once')
    for result in proof['results']:
        for f in result['representation_files'].values():
            if sha(f['path'])!=f['sha256']:raise ValueError('Actual exact representation arrays changed')
    return dict(configuration=characterization.CONFIG,proof_path=str(p),proof_sha256=sha(p),
        guard_path=str(g),guard_sha256=sha(g),matching_profile=PROFILE,
        input_values_only_used_for_registration=True,historical_generator_certified=False)


def registration():
    c=previous.registration();evidence=representation_evidence()
    pins={p['path']:p['sha256'] for p in c['source_receipts']}
    for path in [previous.CONFIG,MODULE.replace('.','/')+'.py',TEST,characterization.CONFIG,
                 'scripts/flow_matching/audit_NASA_split_representation_v1.py',evidence['proof_path'],evidence['guard_path']]:
        pins[path]=sha(ROOT/path)
    c.update(schema_version=3,state_root='experiments/runs/fm_timesnet_all5_declared_CSV_label_control_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/timesnet_all5_declared_CSV_label_control_v1',
        declared_representation_control=dict(channel='G-7',profile=PROFILE,other_channels='direct_float64'),
        representation_evidence=evidence,old_direct_float64_gate_changed=False,
        declared_data_representation_changed=True,model_threshold_and_metric_math_changed=False,
        source_receipts=[dict(path=p,sha256=h) for p,h in sorted(pins.items())])
    c['boundary']+=' Separate declared-data-representation experiment: G7 fixed exact CSV/default-C transform,other50raw arrays direct;old strict-float results remain. No numerical tolerance or score/label-performance choice.'
    return c


def mapped_labels(values,timestamps,raw,metadata_rows):
    direct,_=previous.original.mapped_labels(values,timestamps,raw,metadata_rows)
    channel=metadata_rows[0]['chan_id']
    profile=PROFILE if channel=='G-7' else 'direct_float64'
    represented=characterization.representation(raw,profile)
    evidence,labels=previous.original.mapped_labels(values,timestamps,represented,metadata_rows)
    evidence.update(declared_representation_profile=profile,old_direct_mapping_proven=direct['full_mapping_proven'],
        old_direct_mapping_result_unchanged=True,representation_selected_by_model_performance=False,
        historical_generator_certified=False)
    return evidence,labels


def private_module():
    module=previous.private_module()
    module.MODULE,module.CONFIG,module.TEST=MODULE,CONFIG,TEST
    module.registration=registration;module.mapped_labels=mapped_labels
    return module


def verify(config):
    if config!=registration():raise ValueError('Declared exact representation only,all51entities/all5models/all4label policies required')
    return config['jobs']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true')
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--worker',action='store_true');parser.add_argument('--seed',type=int);args=parser.parse_args()
    if args.register:previous.original.write(ROOT/CONFIG,registration());print(json.dumps(dict(configuration=CONFIG,models=5,label_policies=4)));return
    config=read(ROOT/CONFIG);jobs=verify(config);private=private_module()
    if args.probe_only:print(json.dumps(dict(passed=True,models=5,entities=51,policies=4,
        old_direct_float_gate_changed=False,model_math_changed=False,numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)))
    elif args.worker:
        job=next(j for j in jobs if j['seed']==args.seed);private.worker(config,job,Path(config['artifact_root'])/job['id'])
    else:private.controller(config,jobs)


if __name__=='__main__':main()
