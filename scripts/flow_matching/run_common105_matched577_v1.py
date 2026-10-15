"""Matched common105 cohort:577 new +5 registered DCD +291 unchanged SMD.

Changed SMAP/MSL fit inputs require every associated global model to be fitted
on the new protocol. SMD is referenced through audited all-array equivalence;
original artifacts are never rewritten or counted as independent repetitions.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from types import FunctionType
from scripts.flow_matching import run_original_full_raw873_experiments_v2 as original
from scripts.flow_matching import run_full_smap_DCD105_min105_controls_v1 as dcd

ROOT,read,sha,write=original.ROOT,original.read,original.sha,original.write
MODULE='scripts.flow_matching.run_common105_matched577_v1'
CONFIG='configs/reproducibility/fm_common105_matched577_full873_cohort.2026-10-16_v1.json'
TEST='tests/test_fm_common105_matched577_full873_cohort_v1.py'
COUNTS=dict(objectives=80,neural=340,baselines=65,entropic=30,reflow=40,tranad64=10,moment=12)


def source_material():
    c=read(ROOT/original.CONFIG);p=Path(c['artifact_root'])/'materialization.json';m=read(p)
    if not m['passed'] or m['configuration_sha256']!=sha(ROOT/original.CONFIG) or m['required_cases']!=873:
        raise ValueError('Actual complete873original frozen queues required')
    return c,p,m


def dcd_material():
    c=read(ROOT/dcd.CONFIG);p=Path(c['artifact_root'])/'materialization.json';m=read(p)
    if not m['passed'] or m['configuration_sha256']!=sha(ROOT/dcd.CONFIG) or sha(m['queue'])!=m['queue_sha256']:
        raise ValueError('Actual registered5DCD common105 source queue required')
    q=read(m['queue'])
    if len(q['jobs'])!=5 or len({j['common105_original_source_id'] for j in q['jobs']})!=5:
        raise ValueError('All5distinct DCD original source identities required')
    return c,p,m,q


def registration():
    old,p,m=source_material();dc,dp,dm,dq=dcd_material()
    excluded={j['common105_original_source_id'] for j in dq['jobs']};families=[]
    pins={original.CONFIG,dcd.CONFIG,dcd.prep.CONFIG,str(p),str(dp),dm['queue'],
        MODULE.replace('.','/')+'.py',TEST,dcd.MODULE.replace('.','/')+'.py',
        'scripts/flow_matching/run_original_full_raw873_experiments_v2.py',
        'scripts/flow_matching/prepare_original_full_entity_tsad_min105_v3.py'}
    pins.update(r['path'] for r in old['source_receipts']);pins.update(r['path'] for r in dc['source_receipts'])
    for f in old['families']:
        item=next(q for q in m['queues'] if q['family']==f['name'])
        if sha(item['path'])!=item['sha256']:raise ValueError('Original actual source queue changed')
        queue=read(item['path']);jobs=[j for j in queue['jobs'] if j['dataset'] in ['smap','msl'] and j['id'] not in excluded]
        if len(jobs)!=COUNTS[f['name']]:raise ValueError('Every changed-dataset original method/seed required')
        pins.add(item['path'])
        families.append(dict(name=f['name'],source_queue=item['path'],source_queue_sha256=item['sha256'],
            selected_source_ids=[j['id'] for j in jobs],required_cases=len(jobs),controller_source=f['controller_source']))
    return dict(schema_version=1,full_entity_preparation=dcd.prep.CONFIG,families=families,
        required_new_executions=577,required_referenced_common105_DCD_cases=5,required_unchanged_SMD_references=291,
        required_comparison_cases=873,required_comparison_label_views=2037,
        required_new_per_dataset=dict(smap=286,msl=291),required_cohort_per_dataset=dict(smap=291,msl=291,smd=291),
        CPU_settings=old['CPU_settings'],CPU_resource_lock=old['CPU_resource_lock'],
        GPU_settings=old['GPU_settings'],GPU_resource_lock=old['GPU_resource_lock'],
        existing_DCD_source_configuration=dcd.CONFIG,existing_DCD_materialization=str(dp),
        original_full_source_configuration=original.CONFIG,original_full_materialization=str(p),
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(pins)],
        controller_python=sys.executable,
        artifact_root='F:/aicoding/IIA_Data/experiments/common105_matched577_v1',
        state_root='experiments/runs/fm_common105_matched577_v1',original_scope=old['original_scope'],
        boundary='Full873common105 matched comparison coverage,not a reduced comparator list. '
            '577newSMAP/MSL executions plus5alreadyregistered DCD105 controls and291original SMD references. '
            'All SMAP/MSL globallyfitted models rerun because D-12/T-9 fit inputs/statistics changed. '
            'SMD references require actual28entity all-role/label byte-equivalence proof and immutable bothdata bindings; '
            'original SMD execution/result protocol remains V2,never rebound or counted as new independent training. '
            'Original MOMENT specialized observed-edge-fill worker preserved;native-masked variant remains separate. '
            'Every model,epoch,batch,width,seed,loss,precision,context,MC,solver,evaluation andauthorfidelitylimit retained. '
            'All original CPU/GPU memory/VRAM/UUID/phasegates/sharedlocks/yield retained. '
            'No existing source/native/TAB controller modification,stop,retry or overwriting rawdata. '
            'New scalar/native/alllabel/TAB and pairedtrainingseed reviews required before performance comparison. '
            'All original45papers/681methodrecords/273datatasks/702scalartargets and other jobs remain.')


def verify(c):
    if c!=registration():raise ValueError('All577new+5DCD+291SMD references and original scientific/resource contracts required')
    return c['families']


def SMD_equivalence(new,old,cross):
    if (not cross['passed'] or cross['previous_binding']!=old or cross['all_test_label_overlays_verified']!=298
            or set(cross['changed_entities'])!=set(dcd.prep.CHANGED)):
        raise ValueError('Actual complete109 cross-version evidence and unchanged original data required')
    before,after=old['records']['smd'],new['records']['smd']
    a={e['entity_id']:e for e in before['entities']};b={e['entity_id']:e for e in after['entities']}
    if len(a)!=28 or set(a)!=set(b):raise ValueError('All28original SMD entities required')
    evidence={r['entity_id']:r for r in cross['cases'] if r['entity_id'].startswith('SMD_')}
    if set(evidence)!=set(a):raise ValueError('All28actual SMD byte comparisons required')
    for entity in a:
        row=evidence[entity]
        if (row['split_changed'] or not row['all_unaffected_coordinates_byte_equal'] or not row['test_labels_byte_equal']
                or any(a[entity][k]!=b[entity][k] for k in ['array_prefix','fit_interval','purged_interval','validation_interval',
                    'test_points','features','mean','std','median','bootstrap_group'])):
            raise ValueError('Every SMD role,group,normalizer and target must be unchanged')
    if set(before['labels_only_controls'])!=set(after['labels_only_controls']):raise ValueError('All SMD label policies required')
    return dict(passed=True,entities=28,old_input_sha256=before['input_sha256'],new_input_sha256=after['input_sha256'],
        every_model_input_and_label_byte_equivalent=True,
        source_results_rebound_or_rewritten=False,new_independent_training_repetitions=0)


def ready_data(c):
    new=dcd.binding(c);old=original.ready_data(read(ROOT/original.CONFIG))
    if new is None or old is None:return None
    report=Path(read(ROOT/dcd.prep.CONFIG)['artifact_root'])/dcd.prep.REPORT
    equivalence=SMD_equivalence(new,old,read(report)['cross_version'])
    return dict(**new,original_full_binding=old,SMD_input_equivalence=equivalence,
        cross_version_report=str(report),cross_version_report_sha256=sha(report))


def selected(source,family):
    jobs=[j for j in source['jobs'] if j['id'] in family['selected_source_ids']]
    if [j['id'] for j in jobs]!=family['selected_source_ids'] or len(jobs)!=family['required_cases']:
        raise ValueError('Every explicitly registered changed-input case required')
    return jobs


def clone_queue(c,family,binding):
    source=read(family['source_queue']);queue=deepcopy(source);jobs=[]
    for old in selected(source,family):
        job=deepcopy(old);identity=old['full_raw_source_job_id']+'__common105_v1';record=binding['records'][old['dataset']]
        job.update(id=identity,dataset_manifest_path=binding['manifest'],
            output_directory=c['artifact_root']+'/'+family['name']+'/jobs/'+identity,
            full_raw_data_variant='original_full109_entity_fit_only_min105_v3',common105_original_full_source_id=old['id'])
        pins={p['path']:p['sha256'] for p in old['frozen_files']}
        pins.update({binding['manifest']:binding['manifest_sha256'],record['input_npz']:record['input_sha256'],CONFIG:sha(ROOT/CONFIG)})
        job['frozen_files']=[dict(path=p,sha256=h) for p,h in sorted(pins.items())];jobs.append(job)
    queue.update(jobs=jobs,state_root=c['artifact_root']+'/'+family['name']+'/source_controller',
        full_raw_data_manifest_sha256=binding['manifest_sha256'],
        common105_ancestral_data_manifest_sha256=source['full_raw_data_manifest_sha256'],
        boundary=source['boundary']+'; New common105 matched preprocessing control;original artifacts/protocols retained.')
    validate_clone(source,queue,family,binding)
    return queue


def validate_clone(source,queue,family,binding):
    expected=selected(source,family);actual=queue['jobs'];c=read(ROOT/CONFIG)
    if len(actual)!=family['required_cases']:raise ValueError('Every matched method/seed required')
    allowed={'id','dataset_manifest_path','output_directory','frozen_files','full_raw_data_variant','common105_original_full_source_id'}
    for a,b in zip(expected,actual):
        if (b['id']!=a['full_raw_source_job_id']+'__common105_v1' or b['common105_original_full_source_id']!=a['id']
                or b['full_raw_data_variant']!='original_full109_entity_fit_only_min105_v3'
                or b['dataset_manifest_path']!=binding['manifest']
                or b['output_directory']!=c['artifact_root']+'/'+family['name']+'/jobs/'+b['id']
                or {k:v for k,v in a.items() if k not in allowed}!={k:v for k,v in b.items() if k not in allowed}):
            raise ValueError('Only new data/output/explicit provenance may differ;all scientific budgets required')
        pins={p['path']:p['sha256'] for p in b['frozen_files']};record=binding['records'][b['dataset']]
        needed={p['path']:p['sha256'] for p in a['frozen_files']}
        needed.update({binding['manifest']:binding['manifest_sha256'],record['input_npz']:record['input_sha256'],CONFIG:sha(ROOT/CONFIG)})
        if pins!=needed:raise ValueError('Every old frozen receipt and every new input must remain pinned')
    if (queue['full_raw_data_manifest_sha256']!=binding['manifest_sha256']
            or queue['common105_ancestral_data_manifest_sha256']!=source['full_raw_data_manifest_sha256']):
        raise ValueError('New actual data hash and unchanged ancestral data hash required')
    if set(queue)!=set(source)|{'common105_ancestral_data_manifest_sha256'}:raise ValueError('Only explicit additional data provenance allowed')
    for k,v in source.items():
        if k not in ['jobs','state_root','boundary','full_raw_data_manifest_sha256'] and queue.get(k)!=v:
            raise ValueError('Original source controller/environment/evaluation/phasegates required')


def cohort(c,binding,queues):
    _,_,old=source_material();dc,_,dm,dq=dcd_material()
    DCD={j['common105_original_source_id']:j for j in dq['jobs']}
    fresh={j['common105_original_full_source_id']:(q,j) for q in queues for j in read(q['path'])['jobs']}
    records=[]
    for old_q in old['queues']:
        for job in read(old_q['path'])['jobs']:
            identity=job['id'];family=old_q['family']
            if identity in fresh:
                q,new=fresh[identity];mode='new_common105_execution';path=q['path'];digest=q['sha256']
                guard=ROOT/c['state_root']/'families'/family/'receipts'/(new['id']+'.json')
            elif identity in DCD:
                new=DCD[identity];mode='existing_common105_DCD_execution';path=dm['queue'];digest=dm['queue_sha256']
                guard=ROOT/dc['state_root']/'receipts'/(new['id']+'.json')
            elif job['dataset']=='smd':
                new=job;mode='unchanged_SMD_original_execution';path=old_q['path'];digest=old_q['sha256']
                source=read(ROOT/original.CONFIG);guard=ROOT/source['state_root']/'families'/family/'receipts'/(new['id']+'.json')
            else:raise ValueError('No changed-input original model may be reused')
            records.append(dict(canonical_source_id=job['full_raw_source_job_id'],original_full_source_id=identity,
                actual_source_case_id=new['id'],family=family,dataset=job['dataset'],seed=job['seed'],window=job['window'],
                mode=mode,queue_path=path,queue_sha256=digest,source_guard=str(guard),
                source_dataset_manifest=new['dataset_manifest_path'],original_result_rebound=False,
                required_label_policies=list(binding['records'][job['dataset']]['labels_only_controls'])))
    counts=Counter(r['mode'] for r in records);datasets=Counter(r['dataset'] for r in records)
    if (len(records)!=873 or len({r['canonical_source_id'] for r in records})!=873
            or dict(counts)!=dict(new_common105_execution=577,existing_common105_DCD_execution=5,unchanged_SMD_original_execution=291)
            or dict(datasets)!=c['required_cohort_per_dataset'] or sum(len(r['required_label_policies']) for r in records)!=2037):
        raise ValueError('Entire873method/dataset/seed cohort and2037labels required exactly once')
    return records


def materialize(c,binding):
    output=Path(c['artifact_root']);output.mkdir(parents=True,exist_ok=False);queues=[]
    for family in c['families']:
        queue=clone_queue(c,family,binding);path=output/'queues'/(family['name']+'.json');write(path,queue)
        queues.append(dict(family=family['name'],path=str(path),sha256=sha(path),jobs=len(queue['jobs'])))
    records=cohort(c,binding,queues)
    proof=dict(passed=True,configuration_sha256=sha(ROOT/CONFIG),full_data_binding=binding,queues=queues,
        required_new_executions=577,required_comparison_cases=873,required_comparison_label_views=2037,
        cohort=records,new_training_completed=False,independent_native_and_all_metric_reviews_complete=False,
        all_original_experiments_complete=False)
    write(output/'materialization.json',proof);return proof


def verify_materialization(c):
    proof=read(Path(c['artifact_root'])/'materialization.json');binding=ready_data(c)
    if (binding is None or not proof['passed'] or proof['configuration_sha256']!=sha(ROOT/CONFIG)
            or proof['full_data_binding']!=binding or proof['required_new_executions']!=577):
        raise ValueError('Actual common105/SMD equivalence data and frozen577queues required')
    for q in proof['queues']:
        if sha(q['path'])!=q['sha256']:raise ValueError('Frozen matched source queue changed')
    if proof['cohort']!=cohort(c,binding,proof['queues']):raise ValueError('All873matched references required')
    return proof


def family_controller(c,name):
    body=FunctionType(original.family_controller.__code__,dict(original.family_controller.__globals__,CONFIG=CONFIG,
        verify_materialization=verify_materialization,validate_clone=validate_clone))
    return body(c,name)


def controller(c):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    api=install_facades();base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Existing matched577owner preserved')
    if (base/'status.json').exists():owner.close();raise FileExistsError('Preserve prior matched materialization attempt')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),required_new_executions=577,
        required_comparison_cases=873,all_original_experiments_complete=False)
    def update(**kw):state.update(kw,captured_utc=datetime.now(timezone.utc).isoformat());original.update_file(base/'status.json',state)
    try:
        verify(c);binding=ready_data(c)
        if binding is None:raise ValueError('Actual full common105 preparation required')
        materialize(c,binding);verify_materialization(c);handles=[]
        for family in c['families']:
            path=base/'launches'/family['name'];path.mkdir(parents=True,exist_ok=False)
            command=[c['controller_python'],'-B','-X','utf8','-u','-m',MODULE,'--family',family['name']]
            with (path/'stdout.log').open('xb') as out,(path/'stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src')),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            handle=dict(family=family['name'],pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
            write(path/'launch.json',handle);handles.append(handle)
            update(state='launching_original_guarded_matched_families',family_handles=handles)
        update(state='all7original_controllers_launched_actual_models_and_reviews_pending',family_handles=handles)
    finally:owner.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--probe-only',action='store_true')
    p.add_argument('--family',choices=list(COUNTS));a=p.parse_args()
    if a.register:write(ROOT/CONFIG,registration());print(json.dumps(dict(configuration=CONFIG,new_executions=577,matched_cases=873)));return
    c=read(ROOT/CONFIG);verify(c)
    if a.probe_only:print(json.dumps(dict(passed=True,new_executions=577,existing_DCD_controls=5,unchanged_SMD_references=291,
        matched_cases=873,label_views=2037,actual_common105_data_ready=ready_data(c) is not None,
        numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)))
    elif a.family:family_controller(c,a.family)
    else:controller(c)


if __name__=='__main__':main()
