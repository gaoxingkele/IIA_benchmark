"""Full109 raw control with a common105-point validation minimum.

This is a new data protocol, not a relabelling of old trained results. All
original attempts remain immutable. The pinned V2 numerical preparation body
is reused; a separate proof compares every output with the completed V2 data.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import FunctionType

from scripts.flow_matching import prepare_original_full_entity_tsad_v2 as old

ROOT,read,sha,write=old.ROOT,old.read,old.sha,old.write
MODULE='scripts.flow_matching.prepare_original_full_entity_tsad_min105_v3'
CONFIG='configs/reproducibility/fm_original_full109_entity_preparation_min105.2026-10-16_v3.json'
TEST='tests/test_fm_original_full109_entity_preparation_min105_v3.py'
REPORT='full109_preparation_and_mapping_audit_min105_v3.json'
COMPONENT='source_preparation_component.json'
CHANGED={'SMAP_D-12':(107,207,312),'MSL_T-9':(234,334,439)}


def registration():
    c=deepcopy(old.registration())
    c['recipe'].update(minimum_validation_points=105,
        short_training_series='cut=min(floor(.8*N),N-105);fit_end=cut-100;all fit and validation lengths>=105')
    c.update(artifact_root='F:/aicoding/IIA_Data/experiments/original_full109_entity_tsad_min105_v3',
        state_root='experiments/runs/fm_original_full109_entity_preparation_min105_v3',
        previous_complete_preparation=old.CONFIG,
        new_common_split_protocol=True,previous_model_scores_reusable_as_same_protocol=False,
        compatibility=dict(minimum_unmasked_window_supported=105,
            changed_entities=sorted(CHANGED),all109_entities_retained=True,
            MOMENT512_requires_explicit_native_observation_mask=True),
        recovery=dict(previous_configuration=old.CONFIG,
            proven_failure='Geometry audit: SMAP_D-12 validation100 cannot form DCdetector105 window; MOMENT512 also needs native short-context masks',
            change='Common minimum105 validation;fit-only statistics rebuilt;purge100 unchanged;all old artifacts preserved',
            previous_attempt_preserved=True),
        boundary='New shared105 validation protocol for all109original entities. Exactly D-12/T-9 boundaries change; '
            'all107 other entities and every test label remain byte-identical to completed V2. '
            'Both changed entities are re-normalized from their own new fit segment;old trained results are not interchangeable. '
            'No padding of unmasked DCdetector inputs,window reduction,entity removal,test calibration,or original queue mutation. '
            'MOMENT512 mask adapter and full model executions remain separately required. '
            'All original paper/method/data/702scalar targets and author-specific experiments remain.')
    pins={r['path'] for r in c['source_receipts']}
    pins.update([old.CONFIG,MODULE.replace('.','/')+'.py',TEST,
        'scripts/flow_matching/run_original_full_raw873_experiments_v2.py'])
    previous=read(ROOT/old.CONFIG)
    pins.update([str(Path(previous['artifact_root'])/old.REPORT),
        str(ROOT/previous['state_root']/'guard_receipt.json'),
        str(Path(previous['artifact_root'])/'manifest.json')])
    c['source_receipts']=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(pins)]
    return c


def verify(config):
    if config!=registration():raise ValueError('All109entities,105split,original guards and completed V2 bindings required')
    return config['jobs']


def previous_binding():
    from scripts.flow_matching.run_original_full_raw873_experiments_v2 import ready_data
    binding=ready_data(dict(full_entity_preparation=old.CONFIG))
    if binding is None:raise ValueError('Actual complete109 V2 arrays and successful original guard required')
    return binding


def compare_entity(previous,current,old_arrays,new_arrays):
    """Every real coordinate/label participates; changed splits never use V2 scores."""
    import numpy as np
    entity=current['entity_id']
    if entity!=previous['entity_id']:raise ValueError('Original entity identity changed')
    if set(old_arrays)!=set(new_arrays) or set(new_arrays)!={'train','validation','test','labels'}:
        raise ValueError('All source input roles and labels required')
    if not exact_array(old_arrays['labels'],new_arrays['labels']):raise ValueError('Test labels changed')
    if current['purged_interval'][1]-current['purged_interval'][0]!=100:
        raise ValueError('Original100point purge required')
    if min(len(new_arrays[r]) for r in ['train','validation','test'])<105:
        raise ValueError('Every original role must support105points')
    if entity not in CHANGED:
        if any(not exact_array(old_arrays[r],new_arrays[r]) for r in old_arrays):
            raise ValueError('Unchanged107entities must retain every V2 coordinate')
        for key in ['fit_interval','purged_interval','validation_interval','mean','std','median']:
            if current[key]!=previous[key]:raise ValueError('Unaffected entity preparation changed')
    else:
        fit,cut,total=CHANGED[entity]
        if (current['fit_interval']!=[0,fit] or current['purged_interval']!=[fit,cut]
                or current['validation_interval']!=[cut,total]
                or len(old_arrays['train'])!=fit+5 or len(new_arrays['train'])!=fit
                or len(old_arrays['validation'])!=100 or len(new_arrays['validation'])!=105
                or new_arrays['test'].shape!=old_arrays['test'].shape):
            raise ValueError('Only declared D-12/T-9 original row boundaries may change')
    return dict(entity_id=entity,split_changed=entity in CHANGED,
        all_unaffected_coordinates_byte_equal=entity not in CHANGED,
        test_labels_byte_equal=True,fit_points=len(new_arrays['train']),
        validation_points=len(new_arrays['validation']),test_points=len(new_arrays['test']),
        all_roles_support105=True,normalization_rebuilt_from_new_fit_only=True)


def exact_array(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes(order='C')==b.tobytes(order='C')


def cross_version_audit(config,binding,manifest):
    import numpy as np
    cases=[];labels=0
    for new in manifest['datasets']:
        prior=binding['records'][new['id']]
        if sha(prior['input_npz'])!=prior['input_sha256'] or sha(new['input_npz'])!=new['input_sha256']:
            raise ValueError('Prepared data bytes changed')
        before={r['entity_id']:r for r in prior['entities']}
        if set(before)!={r['entity_id'] for r in new['entities']}:raise ValueError('All109original groups required')
        with np.load(prior['input_npz'],allow_pickle=False) as a,np.load(new['input_npz'],allow_pickle=False) as b:
            if set(a.files)!=set(b.files):raise ValueError('Complete array keys required')
            for current in new['entities']:
                previous=before[current['entity_id']];p,q=previous['array_prefix'],current['array_prefix']
                old_arrays={r:a[p+'_'+r] for r in ['train','validation','test','labels']}
                new_arrays={r:b[q+'_'+r] for r in old_arrays}
                cases.append(compare_entity(previous,current,old_arrays,new_arrays))
                directory=Path(config['artifact_root'])/'entities'/current['entity_id']
                for role,bounds in [('train',current['fit_interval']),('validation',current['validation_interval']),('test',[0,len(new_arrays['test'])])]:
                    if not np.array_equal(np.load(directory/(role+'_original_rows.npy'),allow_pickle=False),np.arange(*bounds,dtype=np.int64)):
                        raise ValueError('Full source row mapping must match exactly')
        for policy,control in new['labels_only_controls'].items():
            prior_control=prior['labels_only_controls'][policy]
            if sha(prior_control['path'])!=prior_control['sha256'] or sha(control['path'])!=control['sha256']:
                raise ValueError('Label control file changed')
            with np.load(prior_control['path'],allow_pickle=False) as a,np.load(control['path'],allow_pickle=False) as b:
                if set(a.files)!=set(b.files) or any(not exact_array(a[k],b[k]) for k in a.files):
                    raise ValueError('All7full test label controls must remain identical')
                labels+=len(a.files)
    if len(cases)!=109 or {r['entity_id'] for r in cases if r['split_changed']}!=set(CHANGED) or labels!=298:
        raise ValueError('109entities/exactly2changes/all298entity-label overlays required')
    return dict(passed=True,entities=109,unchanged_entities=107,changed_entities=sorted(CHANGED),
        all_test_label_overlays_verified=labels,cases=cases,
        previous_binding=binding,old_model_outputs_are_same_protocol=False)


def worker(config):
    verify(config);binding=previous_binding()
    body=FunctionType(old.worker.__code__,dict(old.worker.__globals__,CONFIG=CONFIG,REPORT=COMPONENT,verify=verify))
    body(config)
    output=Path(config['artifact_root']);component=read(output/COMPONENT);manifest=read(output/'manifest.json')
    comparison=cross_version_audit(config,binding,manifest)
    for record in manifest['datasets']:record['source_variant']='original_full_entity_fit_only_min105_v3'
    complete=output/'common_min105_manifest.json';write(complete,manifest)
    verify(config)
    report=dict(component,prepared_manifest=str(complete),prepared_manifest_sha256=sha(complete),
        source_component_sha256=sha(output/COMPONENT),cross_version=comparison,
        old_scores_same_protocol_certified=False,all_original_experiments_complete=False)
    write(output/REPORT,report)


def controller(config):
    body=FunctionType(old.controller.__code__,dict(old.controller.__globals__,MODULE=MODULE,CONFIG=CONFIG,REPORT=REPORT))
    return body(config)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register',action='store_true');parser.add_argument('--worker',action='store_true')
    parser.add_argument('--probe-only',action='store_true');args=parser.parse_args()
    if args.register:
        write(ROOT/CONFIG,registration());print(json.dumps(dict(configuration=CONFIG,entities=109)));return
    config=read(ROOT/CONFIG);verify(config)
    if args.probe_only:print(json.dumps(dict(passed=True,entities=109,changed_entities=sorted(CHANGED),
        minimum_validation_points=105,numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)))
    elif args.worker:worker(config)
    else:controller(config)


if __name__=='__main__':main()
