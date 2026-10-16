"""Audit all trained artifacts and compare full paired EB protocol cohorts.

Metadata/streaming-integrity only: never import a model framework, train, infer,
replace an LP result, or turn partial cohorts into performance estimates.
"""
import argparse
import ast
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import time
import zipfile
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,write
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive

MODULE='scripts.flow_matching.collect_trajectorynet_two_protocols_full270_v1'
CONFIG='configs/reproducibility/fm_TrajectoryNet_two_protocols_full270_comparison.v1.json'
QUEUES={
 'author_log':'configs/experiments/fm_TrajectoryNet_EB_native_full180.v1.json',
 'paper_text':'configs/experiments/fm_TrajectoryNet_EB_paper_text_full180.v1.json'}
SELECTIONS=['corrected_validation_minimum','source_last_scheduled_validation']
METRICS=['W1_past','W1_future','W2_past','W2_future']
SEEDS=[42,43,44,45,46]
LEFT=[1,2,3]


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):result.update(chunk)
    return result.hexdigest()


def registration():
    queues={name:read(ROOT/path) for name,path in QUEUES.items()}
    a,b=queues['author_log'],queues['paper_text']
    changed={k for k,v in a['source_parameters'].items() if b['source_parameters'].get(k)!=v}
    if changed!={'nonlinearity','weight_decay'} or a['original_data_sha256']!=b['original_data_sha256']:
        raise ValueError('Compare only the predeclared activation/decay recipes on exactly the same complete raw input')
    if a['full_population_counts']!=b['full_population_counts'] or a['settings']!=b['settings']:
        raise ValueError('Original populations and numerical admission guards must stay identical')
    cases=[]
    for recipe in range(9):
        pairs={name:[j for j in q['jobs'] if j['recipe_index']==recipe] for name,q in queues.items()}
        for jobs in pairs.values():
            if len(jobs)!=15 or {(j['seed'],j['leftout_timepoint']) for j in jobs}!={(s,t) for s in SEEDS for t in LEFT}:
                raise ValueError('All five seeds and all three leftouts must be paired for every recipe')
        cases.append(dict(id=f'recipe{recipe}',recipe_index=recipe,recipe=pairs['author_log'][0]['recipe'],jobs=pairs,
            required_models=30,required_checkpoint_selections=60))
    paths=[MODULE.replace('.','/')+'.py','tests/test_fm_trajectorynet_full270_comparison_v1.py',
        'projects/flow_matching_research/ara/adopted_baselines/trajectorynet_paper_text_vs_logged_recipe.2026-10-16_v1.json']
    for name,q in queues.items():
        paths.append(QUEUES[name]);paths.extend(r['path'] for r in q['source_receipts'])
    return dict(cases=cases,required_comparisons=9,required_models=270,required_checkpoint_selections=540,
        queues={name:dict(path=QUEUES[name],sha256=digest(ROOT/QUEUES[name]),state_root=q['state_root']) for name,q in queues.items()},
        full_population_counts=a['full_population_counts'],original_data=a['original_data'],
        original_data_sha256=a['original_data_sha256'],training_seeds=SEEDS,all_leftouts=LEFT,
        checkpoint_selections=SELECTIONS,metrics=METRICS,primary_metric='W1_past',
        metric_tolerance=a['metric_tolerance'],metadata_publication_stability_seconds=2,
        uncertainty_unit='five independent training seed means across all three leaveouts',
        paired_contrast_direction='paper_text_minus_author_log',individual_activation_decay_effect_identifiable=False,
        fixed_hyperparameters=True,test_coefficient_selection_allowed=False,partial_cohorts_allowed=False,
        source_receipts=[dict(path=p,sha256=digest(ROOT/p)) for p in sorted(set(paths))],
        state_root='experiments/runs/fm_TrajectoryNet_two_protocols_full270_comparison_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/TrajectoryNet_two_protocols_full270_comparison_v1',
        output_root='projects/flow_matching_research/results/2026-10-16/TrajectoryNet_two_protocols_full270_comparisons_v1',
        original_scope=a['original_scope'],pending_energy_models_in_two_protocols=90,
        model_frameworks_imported=False,numerical_inference_performed=False,
        all_original_experiments_complete=False)


def verify(c):
    if c!=registration():raise ValueError('Frozen complete paired protocols and every source binding are required')
    if digest(Path(c['original_data']))!=c['original_data_sha256']:
        raise ValueError('Full original raw input changed')


def npy_identity(path,shape,dtype):
    path=Path(path)
    with path.open('rb') as stream:
        if stream.read(6)!=b'\x93NUMPY':raise ValueError('Complete original numeric NPY required')
        version=tuple(stream.read(2))
        if version not in [(1,0),(2,0),(3,0)]:raise ValueError('Unsupported NPY container')
        size=2 if version==(1,0) else 4
        length=struct.unpack('<H' if size==2 else '<I',stream.read(size))[0]
        if length>2**20:raise ValueError('Unexpected NPY header size')
        header=ast.literal_eval(stream.read(length).decode('utf-8' if version==(3,0) else 'latin1').strip())
        start=stream.tell()
    if header.get('shape')!=tuple(shape) or header.get('descr')!=dtype or not isinstance(header.get('fortran_order'),bool):
        raise ValueError('Every full prediction and truth array must retain shape and original float dtype')
    itemsize={'<f4':4,'<f8':8}[dtype]
    if path.stat().st_size-start!=math.prod(shape)*itemsize:
        raise ValueError('Truncated or extra numeric array bytes')
    return dict(path=str(path),shape=list(shape),dtype=dtype,fortran_order=header['fortran_order'],
        bytes=path.stat().st_size,sha256=digest(path))


def paired_array_identity(source,replay,shape,dtype):
    a,b=npy_identity(source,shape,dtype),npy_identity(replay,shape,dtype)
    if a['sha256']!=b['sha256']:
        raise ValueError('Every original and checkpoint-replayed array must be byte-for-byte identical')
    return dict(source=a,replay=b,complete_file_bytes_identical=True)


def checkpoint_payloads(path):
    # Normalize the ZIP root folder generated from the checkpoint filename.
    # No pickle global or tensor object is ever deserialized or executed.
    records=[];seen=set()
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if member.is_dir():continue
            # ZIP uses '/' even on Windows. Avoid host-path interpretation here.
            relative=member.filename.split('/',1)[-1]
            if '/' not in member.filename or relative in seen or '..' in relative.split('/') or member.file_size>2**20:
                raise ValueError('Require bounded unique original model checkpoint payloads')
            seen.add(relative)
            content=archive.read(member)
            if len(content)!=member.file_size:raise ValueError('Truncated checkpoint ZIP member')
            records.append(dict(member=relative,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
    if 'data.pkl' not in seen or not any(n.startswith('data/') for n in seen):
        raise ValueError('Actual Torch checkpoint storage and metadata members required')
    return sorted(records,key=lambda r:r['member'])


def verify_selected_step(best,periodic):
    left,right=checkpoint_payloads(best),checkpoint_payloads(periodic)
    if left!=right:raise ValueError('Selected checkpoint must have exactly the payload of its claimed original periodic training step')
    return dict(selected=str(best),periodic=str(periodic),normalized_zip_payloads_equal=True,
        payloads=left,pickle_or_model_execution_performed=False)


def near(a,b):
    return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12)


def metric_proof(record,leftout,counts):
    if (set(record['metrics'])!=set(METRICS) or not record['full_population']
            or record['primary_direction']!='past_to_leftout' or record['selected_better_direction']
            or not near(record['primary_W1'],record['metrics']['W1_past']) or len(record['proofs'])!=2):
        raise ValueError('Both complete directional native metrics and fixed primary direction required')
    for index,p in enumerate(record['proofs']):
        direction='future' if index==0 else 'past'
        adjacent=leftout+(1 if index==0 else -1)
        shape=[counts[str(adjacent)],counts[str(leftout)]]
        native=p['native_W2'];raw=p['raw_W1'];independent=p['independent_float64_W1']
        if (p['direction']!=direction or p['shape']!=shape or native['shape']!=shape
                or p['all_cost_entries_checked']!=math.prod(shape)
                or native['result_code']!=1 or p['native_W1_LP_result_code']!=1 or p['independent_W1_LP_result_code']!=1):
            raise ValueError('Full actual exact-LP population dimensions and every cost-entry proof required')
        numbers=[raw,independent,native['reported_W2'],native['transport_objective'],p['maximum_cost_entry_difference'],p['objective_difference'],p['cost_perturbation_bound']]
        if any(not math.isfinite(float(v)) or float(v)<0 for v in numbers):
            raise ValueError('Every native and independent metric/proof scalar must be finite and nonnegative')
        bound=p['maximum_cost_entry_difference']+1e-8*max(1.,abs(raw),abs(independent))
        if (not near(p['cost_perturbation_bound'],bound) or not near(p['objective_difference'],abs(raw-independent))
                or abs(raw-independent)>bound
                or not near(native['reported_W2']**2,native['transport_objective'])
                or not near(record['metrics']['W1_'+direction],raw)
                or not near(record['metrics']['W2_'+direction],native['reported_W2'])):
            raise ValueError('Raw W1, squared-cost W2 and independent transport bound must not be interchanged')


def check_guard(path,report):
    guard=read(path)
    if not guard['passed'] or guard['exit_code']!=0 or guard['resource_guard_aborted'] or guard['report_sha256']!=digest(report):
        raise ValueError('Only a real successful numerical guard with unchanged result payload may contribute')
    return dict(path=str(path),sha256=digest(path),report=str(report),report_sha256=digest(report))


def inspect_model(c,queue,queue_path,job):
    out=Path(job['artifact_directory']);source_path=out/'source_result.json';native_path=out/'native_replay/report.json'
    source,native=read(source_path),read(native_path);key=hashlib.sha256(job['id'].encode()).hexdigest()[:20]
    guards=[check_guard(ROOT/queue['state_root']/'attempts'/(key+'_'+phase)/'guard_receipt.json',path)
        for phase,path in [('train',source_path),('replay',native_path)]]
    for r in [source,native]:
        if not r['passed'] or r['job']!=job or r['queue_sha256']!=digest(ROOT/queue_path):
            raise ValueError('Actual original trained model and replay identities differ')
    if (source['actual_optimizer_updates']!=10000 or native['actual_training_updates']!=10000
            or source['parameters']!=9621 or source['training_loss_calls']!=10000 or source['validation_loss_calls']!=100
            or source['periodic_checkpoints']!=400 or source['raw_input_sha256']!=c['original_data_sha256']
            or source['full_population_counts']!=c['full_population_counts']
            or native['source_result_sha256']!=digest(source_path)):
        raise ValueError('Every full native budget, input and actual source-to-replay lineage is required')
    history=read(out/'full_loss_history.json')
    if ([r['steps'] for r in history['training']]!=list(range(10000))
            or [r['steps'] for r in history['validation']]!=list(range(100,10001,100))
            or any(not math.isfinite(r['loss']) for r in history['training']+history['validation'])):
        raise ValueError('Full10000training and100sampled-validation loss history required')
    minima=[];minimum=float('inf')
    for row in history['validation']:
        if row['loss']<minimum:minimum=row['loss'];minima.append(dict(step=row['steps'],loss=row['loss']))
    if ([(r['step'],r['loss']) for r in history['selected_history']]!=[(r['step'],r['loss']) for r in minima]
            or source['corrected_best_step']!=minima[-1]['step'] or source['corrected_best_validation_loss']!=minimum
            or source['source_final_step']!=10000):
        raise ValueError('Validation-selected step must follow the entire original criterion without test selection')
    expected={f'checkpt-{step:04d}.pth' for step in range(25,10001,25)}
    actual={p.name for p in out.glob('checkpt-*.pth')}
    if actual!=expected:raise ValueError('All400 original periodic checkpoint names and files required')
    periodic=[]
    for name in sorted(expected):
        payloads=checkpoint_payloads(out/name)
        periodic.append(dict(path=str(out/name),sha256=digest(out/name),bytes=(out/name).stat().st_size,
            every_zip_payload_CRC_read=True,payload_manifest_sha256=hashlib.sha256(json.dumps(payloads,sort_keys=True).encode()).hexdigest()))
    lineage=verify_selected_step(out/'checkpt.pth',out/f'checkpt-{source["corrected_best_step"]:04d}.pth')
    if history['selected_history'][-1]['checkpoint_sha256']!=digest(out/'checkpt.pth'):
        raise ValueError('Selected history must bind the actual retained minimum checkpoint')
    for r in source['checkpoints']+source['full_state_files']:
        if digest(Path(r['path']))!=r['sha256']:raise ValueError('Frozen checkpoint or full optimizer/RNG state changed')
    if {r['selection'] for r in source['checkpoints']}!=set(SELECTIONS):
        raise ValueError('Both checkpoint selections required')
    arrays=[]
    for selection in SELECTIONS:
        original=source['source_scores'][selection];result=native['checkpoint_selections'][selection]
        metric_proof(original,job['leftout_timepoint'],c['full_population_counts'])
        metric_proof(result,job['leftout_timepoint'],c['full_population_counts'])
        if not result['all_prediction_and_truth_coordinates_bitwise_equal']:
            raise ValueError('Actual whole-population native checkpoint replay must have passed')
        if any(not math.isclose(original['metrics'][k],result['metrics'][k],
                rel_tol=c['metric_tolerance']['rtol'],abs_tol=c['metric_tolerance']['atol']) for k in METRICS):
            raise ValueError('Full deterministic CPU replay scalars must agree before paired inference')
        source_folder=out/('corrected_best_metrics' if selection==SELECTIONS[0] else 'source_final_metrics')
        replay_folder=out/'native_replay'/selection
        for direction in ['future','past']:
            adjacent=job['leftout_timepoint']+(1 if direction=='future' else -1)
            for suffix,shape,dtype in [('prediction',[c['full_population_counts'][str(adjacent)],5],'<f4'),
                ('native_truth64',[c['full_population_counts'][str(job['leftout_timepoint'])],5],'<f8'),
                ('matched_truth32',[c['full_population_counts'][str(job['leftout_timepoint'])],5],'<f4')]:
                name=direction+'_'+suffix+'.npy'
                arrays.append(dict(selection=selection,role=name,**paired_array_identity(source_folder/name,replay_folder/name,shape,dtype)))
    return dict(job=job,metrics={s:native['checkpoint_selections'][s]['metrics'] for s in SELECTIONS},
        source_and_native_guards=guards,loss_history_sha256=digest(out/'full_loss_history.json'),
        actual_training_updates=10000,corrected_selected_step=source['corrected_best_step'],
        selected_checkpoint_lineage=lineage,all400_periodic_checkpoints=periodic,complete_arrays=arrays)


def uncertainty(values):
    if len(values)!=5 or any(not math.isfinite(v) for v in values):raise ValueError('All five independent finite seed values required')
    mean,sd=statistics.mean(values),statistics.stdev(values);half=2.7764451051977987*sd/math.sqrt(5)
    return dict(mean=mean,SD=sd,descriptive_df4_t95=[mean-half,mean+half])


def summaries(records):
    expected={(s,t) for s in SEEDS for t in LEFT}
    if len(records)!=30:raise ValueError('All30 paired original models required')
    indexed={}
    for profile in QUEUES:
        rows=[r for r in records if r['profile']==profile]
        if len(rows)!=15 or {(r['seed'],r['leftout']) for r in rows}!=expected:
            raise ValueError('No partial-seed or partial-leaveout performance row is allowed')
        indexed[profile]={(r['seed'],r['leftout']):r for r in rows}
    cohorts,contrasts={},{}
    for profile,rows in indexed.items():
        cohorts[profile]={}
        for selection in SELECTIONS:
            cohorts[profile][selection]={}
            for metric in METRICS:
                values=[rows[s,t]['metrics'][selection][metric] for s in SEEDS for t in LEFT]
                means=[statistics.mean(rows[s,t]['metrics'][selection][metric] for t in LEFT) for s in SEEDS]
                cohorts[profile][selection][metric]=dict(**uncertainty(means),seed_leaveout_means=dict(zip(map(str,SEEDS),means)),
                    separate_all15_model_SD=statistics.stdev(values),
                    per_leftout={str(t):uncertainty([rows[s,t]['metrics'][selection][metric] for s in SEEDS]) for t in LEFT})
    for selection in SELECTIONS:
        for metric in METRICS:
            values=[statistics.mean(indexed['paper_text'][s,t]['metrics'][selection][metric]-indexed['author_log'][s,t]['metrics'][selection][metric] for t in LEFT) for s in SEEDS]
            contrasts['paper_text_minus_author_log__'+selection+'__'+metric]=dict(**uncertainty(values),paired_seed_differences=dict(zip(map(str,SEEDS),values)))
    for profile in QUEUES:
        for metric in METRICS:
            values=[statistics.mean(indexed[profile][s,t]['metrics'][SELECTIONS[0]][metric]-indexed[profile][s,t]['metrics'][SELECTIONS[1]][metric] for t in LEFT) for s in SEEDS]
            contrasts[profile+'__corrected_minimum_minus_source_final__'+metric]=dict(**uncertainty(values),paired_seed_differences=dict(zip(map(str,SEEDS),values)))
    return dict(full_models=30,independent_training_seeds=SEEDS,all_leftouts=LEFT,
        cohorts=cohorts,paired_contrasts=contrasts,uncertainty_unit='five seed means across all three leaveouts',
        activation_and_decay_effects_individually_identifiable=False,test_parameter_selection=False,
        formal_hypothesis_testing_or_causal_claim=False,strict_TAB_result=False,all_original_experiments_complete=False)


def close_case(c,case):
    queues={name:read(ROOT/entry['path']) for name,entry in c['queues'].items()}
    row_paths={name:ROOT/q['state_root']/'complete_rows'/f'recipe{case["recipe_index"]}.json' for name,q in queues.items()}
    if not all(p.exists() for p in row_paths.values()):return None
    if any(time.time()-p.stat().st_mtime<c['metadata_publication_stability_seconds'] for p in row_paths.values()):return None
    raw=Path(c['artifact_root'])/case['id'];raw.mkdir(parents=True,exist_ok=False)
    records=[];manifests=[];source_rows=[];probe_bindings=[];author_bindings=[]
    for profile,q in queues.items():
        row=read(row_paths[profile])
        if row['full_models']!=15 or row['queue_sha256']!=c['queues'][profile]['sha256'] or row['recipe']!=case['recipe']:
            raise ValueError('Both complete native numerical cohorts must bind the same grid recipe')
        probe=Path(q['artifact_root'])/'native_full27_probe/report.json'
        probe_guard=check_guard(ROOT/q['state_root']/'attempts/full27_probe/guard_receipt.json',probe)
        pr=read(probe)
        if (not pr['passed'] or pr['configuration_sha256']!=c['queues'][profile]['sha256']
                or len(pr['records'])!=27 or pr['optimizer_updates']!=0
                or {(r['recipe_index'],r['leftout_timepoint']) for r in pr['records']}!={(i,t) for i in range(9) for t in LEFT}
                or {r['leftout_timepoint'] for r in pr['full_population_native_evaluations']}!=set(LEFT)):
            raise ValueError('Full27 source loss/backward API preflight must have passed without being a benchmark')
        for evaluation in pr['full_population_native_evaluations']:
            metric_proof(evaluation,evaluation['leftout_timepoint'],c['full_population_counts'])
        probe_bindings.append(dict(profile=profile,guard=probe_guard))
        author=Path(q['author_replay_report']);author_guard=check_guard(ROOT/q['author_replay_guard'],author)
        ar=read(author)
        if not ar['passed'] or ar['parameters']!=9621 or ar['full_population_counts']!=c['full_population_counts']:
            raise ValueError('Actual original full author checkpoint replay must have passed')
        author_bindings.append(dict(profile=profile,guard=author_guard))
        source_rows.append(dict(profile=profile,path=str(row_paths[profile]),sha256=digest(row_paths[profile])))
        for job in case['jobs'][profile]:
            record=inspect_model(c,q,c['queues'][profile]['path'],job)
            manifests.append(record)
            records.append(dict(profile=profile,seed=job['seed'],leftout=job['leftout_timepoint'],metrics=record['metrics']))
    # The same raw population and order must be retained across both protocols.
    truth={}
    for record in manifests:
        for a in record['complete_arrays']:
            if 'truth' not in a['role']:continue
            key=(record['job']['leftout_timepoint'],a['role'].split('_',1)[1])
            value=a['source']['sha256']
            if key in truth and truth[key]!=value:raise ValueError('Paired protocols/seeds must use identical full target arrays')
            truth[key]=value
    report=summaries(records)
    manifest=raw/'all_model_array_checkpoint_and_guard_bindings.json'
    exclusive(manifest,dict(models=manifests,source_complete_rows=source_rows,
        original_author_checkpoint_bindings=author_bindings,full27_probe_bindings=probe_bindings,
        all_original_experiments_complete=False))
    audit=read(ROOT/'projects/flow_matching_research/ara/adopted_baselines/trajectorynet_paper_text_vs_logged_recipe.2026-10-16_v1.json')
    report.update(passed=True,case=case['id'],recipe=case['recipe'],configuration_sha256=digest(ROOT/CONFIG),
        complete_artifact_manifest=dict(path=str(manifest),sha256=digest(manifest)),
        original_paper_reference_EMD=audit['active_Table3'][case['recipe']['family']],
        paper_metric_direction_and_selected_coefficients_certified=False,
        W1_and_W2_not_interchanged=True,model_frameworks_imported=False,numerical_inference_performed=False)
    target=ROOT/c['output_root'];target.mkdir(parents=True,exist_ok=True)
    exclusive(target/(case['id']+'.json'),report)
    with (target/(case['id']+'.csv')).open('x',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['recipe','profile','checkpoint_selection','metric','mean','five_seed_SD','t95_low','t95_high','all15_model_SD'])
        for profile in QUEUES:
            for selection in SELECTIONS:
                for metric in METRICS:
                    r=report['cohorts'][profile][selection][metric]
                    writer.writerow([case['id'],profile,selection,metric,r['mean'],r['SD'],*r['descriptive_df4_t95'],r['separate_all15_model_SD']])
    return report


def controller(c):
    import os
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    api=install_facades();base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None or (base/'status.json').exists():
        if owner:owner.close()
        raise FileExistsError('Never overwrite or restart an earlier complete-cohort metadata review')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),required_comparisons=9,required_models=270,
        completed_cases=[],failed_preserved={},numerical_inference_performed=False,all_original_experiments_complete=False)
    def update(value):state.update(value,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        while len(state['completed_cases'])+len(state['failed_preserved'])<9:
            for case in c['cases']:
                if case['id'] in state['completed_cases'] or case['id'] in state['failed_preserved']:continue
                try:
                    result=close_case(c,case)
                    if result is not None:state['completed_cases'].append(case['id'])
                except Exception as error:state['failed_preserved'][case['id']]=repr(error)
            update(dict(state='waiting_complete30_model_cohorts_and_unchanged_full_artifacts'))
            time.sleep(10)
        update(dict(state='all9_paired_reviews_terminal_remaining_original_experiments_and_energy_conditions_still_pending'))
    finally:owner.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');a=p.parse_args()
    if a.register:exclusive(ROOT/CONFIG,registration());return
    c=read(ROOT/CONFIG);verify(c);controller(c)


if __name__=='__main__':main()
