"""Independent full saved-score audit for every original industrial CPU slot.

No model/NumPy/sklearn imports, no historical resource certification, no new
training credit. Source checkpoints are hashed, not replayed by this module.
"""
import argparse
import array
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
import zipfile

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write
from scripts.flow_matching.audit_industrial_SB_extended_full_replay_v1 import roles, strict_metrics, SUFFIXES

MODULE = 'scripts.flow_matching.audit_industrial_full550_independent_metrics_v1'
TEST = 'tests/test_fm_industrial_full550_independent_metrics_v1.py'
CONFIG = 'configs/reproducibility/fm_industrial_full550_independent_metrics.v1.json'
QUEUE = 'configs/experiments/fm_industrial_tsad_cpu_queue.v1.json'
RAW_AUDIT = 'configs/reproducibility/fm_industrial_full_raw_preprocessing.2026-10-16_v1.json'


def registration():
    q = read(ROOT/QUEUE)
    raw = read(ROOT/RAW_AUDIT)
    manifest = q['jobs'][0]['dataset_manifest_path']
    files = [QUEUE, RAW_AUDIT, manifest, raw['report_path'], MODULE.replace('.','/')+'.py', TEST,
        'scripts/flow_matching/audit_industrial_SB_extended_full_replay_v1.py',
        'scripts/flow_matching/audit_crossad_full139_independent_metrics_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py']
    return dict(queue=QUEUE, manifest=manifest, required_original_CPU_slots=550,
        original_industrial_GPU_scope_unchanged=850,
        original_global_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273,paper_scalar_targets=702),
        raw_preprocessing_audit=raw['report_path'],
        state_root='experiments/runs/fm_industrial_full550_independent_metrics_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/industrial_full550_independent_metrics_v1',
        resource_lock='experiments/runs/fm_bounded_transport_math_diagnostic_cpu.lock',
        settings=dict(minimum_available_memory_bytes=12*2**30,minimum_commit_headroom_bytes=22*2**30,
            emergency_commit_headroom_bytes=16*2**30,maximum_worker_rss_bytes=2**30,maximum_worker_private_bytes=2**30),
        numerical_metric_tolerance=dict(rtol=2e-6,atol=2e-6),
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in files],
        source_checkpoint_neural_replay=False,source_historical_resource_guard_certified=False,
        independent_bootstrap=False,independent_VUS=False,independent_affiliation=False,
        new_training_seed_slots=0,new_optimizer_updates=0,new_neural_forwards=0,
        new_complete_model_results=0,full_TAB_equivalence=False,paper_equivalence=False,
        citations=read(ROOT/'configs/experiments/fm_industrial_tsad_execution.v1.json')['citations'],
        all_original_experiments_complete=False)


def verify(c):
    if c != registration(): raise ValueError('Original full550 scope, frozen input audit and bounded metric guards changed')
    q = read(ROOT/c['queue'])
    if len(q['jobs']) != 550 or len({j['id'] for j in q['jobs']}) != 550:
        raise ValueError('Every original industrial CPU condition remains required')
    return q, read(ROOT/c['manifest'])


def npy(stream):
    """Bounded pickle-free NPY reader; validate full payload before decoding."""
    if stream.read(6) != b'\x93NUMPY': raise ValueError('NPY array required')
    version = stream.read(2)
    width = {b'\x01\x00':2,b'\x02\x00':4,b'\x03\x00':4}.get(version)
    if width is None: raise ValueError('Unsupported NPY version')
    count = stream.read(width)
    if len(count) != width: raise ValueError('Truncated NPY header length')
    size = int.from_bytes(count,'little')
    if not 0 < size <= 65536: raise ValueError('Bounded NPY header required')
    text = stream.read(size)
    if len(text) != size: raise ValueError('Truncated NPY header')
    header = ast.literal_eval(text.decode('utf-8' if version==b'\x03\x00' else 'latin1'))
    if (set(header) != {'descr','fortran_order','shape'} or header['fortran_order'] is not False
            or not isinstance(header['shape'],tuple) or not 1 <= len(header['shape']) <= 2
            or any(type(n) is not int or n <= 0 for n in header['shape'])):
        raise ValueError('Complete C-order numeric/Unicode array required')
    dtype = header['descr']
    itemsize = {'<f8':8,'<f4':4,'<i8':8,'|u1':1}.get(dtype)
    if itemsize is None:
        match = re.fullmatch(r'<U([1-9][0-9]{0,3})',dtype)
        if not match: raise ValueError('Object, structured, endian-changed or unknown dtype forbidden')
        itemsize = 4*int(match[1])
    expected = math.prod(header['shape'])*itemsize
    if expected > 64*2**20: raise ValueError('Per-array bounded diagnostic memory exceeded')
    payload = stream.read(expected+1)
    if len(payload) != expected or stream.read(1): raise ValueError('Truncated or trailing NPY payload')
    return header, payload


def archive_array(archive, name):
    with archive.open(name+'.npy') as stream: return npy(stream)


def vector(header, payload, count, dtype):
    if header != dict(descr=dtype,fortran_order=False,shape=(count,)):
        raise ValueError('Full original vector dtype and shape required')
    code = {'<f8':'d','|u1':'B'}[dtype]
    result = array.array(code); result.frombytes(payload)
    if sys.byteorder != 'little' and result.itemsize > 1: result.byteswap()
    if (dtype=='<f8' and not all(map(math.isfinite,result))) or (dtype=='|u1' and any(x not in (0,1) for x in result)):
        raise ValueError('Every original score finite and label binary required')
    return result


def expected_budget(dataset, window):
    budget = []
    for g in dataset['training_groups']:
        n = g['points']
        if n < window: raise ValueError('An original fit group is shorter than its frozen window')
        windows = (n-window)//window+1+int((n-window)%window!=0)
        budget.append(dict(group_id=g['group_id'],source_interval=g['source_interval'],points=n,windows=windows))
    if sum(x['points'] for x in budget) != dataset['scaler']['fit_points']:
        raise ValueError('Unique full native training coordinates required')
    return budget


def audit(c, q, manifest, job):
    for p in job['frozen_files']:
        if sha(ROOT/p['path']) != p['sha256']: raise ValueError('Frozen original source changed: '+p['path'])
    output = ROOT/job['output_directory']
    paths = dict(result=output/'result.json',checkpoint=output/'model.pt',scores=output/'scores.npz')
    before = {k:sha(p) for k,p in paths.items()}
    source = read(paths['result'])
    if (source['id'] != job['id'] or source['job'] != job or source['status'] != 'completed'
            or source['checkpoint_sha256'] != before['checkpoint'] or source['scores_sha256'] != before['scores']
            or source['experiment_sha256'] != hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()):
        raise ValueError('Complete original job and model/score artifact identity required')
    dataset = next(d for d in manifest['datasets'] if d['id']==job['dataset'])
    if sha(ROOT/dataset['input_npz']) != dataset['input_sha256']: raise ValueError('Complete frozen industrial input changed')
    for p in dataset['source_files']:
        if sha(ROOT/p['path']) != p['sha256']: raise ValueError('Original raw industrial data changed')
    budget = expected_budget(dataset,job['window'])
    if budget != source['fit_group_budget'] or sum(x['windows'] for x in budget) != source['unique_fit_windows']:
        raise ValueError('Original distinct fit groups/windows changed')
    configuration = read(ROOT/job['model_config'])
    p = configuration['parameters']; epochs = p['epochs']+(p.get('reflow_stages',1)-1)*p.get('reflow_epochs',0)
    losses = source['training_losses']
    if (source['epoch_audit']['declared_epochs'] != epochs
            or losses is not None and (len(losses)!=epochs or not all(map(math.isfinite,losses)))):
        raise ValueError('Full declared and available observed epoch history required')
    payload, bindings, coverage = {}, {}, []
    with zipfile.ZipFile(ROOT/dataset['input_npz']) as raw, zipfile.ZipFile(paths['scores']) as scores:
        if len(scores.namelist()) != len(set(scores.namelist())) or len(raw.namelist()) != len(set(raw.namelist())):
            raise ValueError('Duplicate ZIP array names forbidden')
        expected = set()
        for role,g in roles(dataset):
            prefix = g['array_prefix']; key = prefix+'_'+role; expected.add(key+'.npy')
            h,b = archive_array(scores,key)
            payload[key] = vector(h,b,g['points'],'<f8')
            bindings[key] = dict(dtype=h['descr'],shape=list(h['shape']),sha256=hashlib.sha256(b).hexdigest())
            input_name = prefix+('_test' if role=='test' else '_values')
            ih,ib = archive_array(raw,input_name)
            if ih != dict(descr='<f4',fortran_order=False,shape=(g['points'],g['features'])):
                raise ValueError('Full original normalized feature matrix required')
            for suffix in SUFFIXES:
                name = prefix+suffix
                if name+'.npy' in raw.namelist():
                    expected.add(name+'.npy'); actual = archive_array(scores,name); original = archive_array(raw,name)
                    if actual != original: raise ValueError('Original full labels/timestamps/source rows/faults differ: '+name)
                    mh,mb = actual
                    if mh['shape'] != (g['points'],): raise ValueError('Every metadata coordinate must align')
                    bindings[name] = dict(dtype=mh['descr'],shape=list(mh['shape']),sha256=hashlib.sha256(mb).hexdigest())
                    if suffix=='_labels': payload[name] = vector(mh,mb,g['points'],'|u1')
            if role=='test' and (prefix+'_labels' not in payload or g['points']!=g['test_points']):
                raise ValueError('Complete original test labels and native coverage required')
            coverage.append(dict(role=role,group=g['group_id'],points=g['points'],features=g['features']))
        if set(scores.namelist()) != expected: raise ValueError('Exact complete score/metadata archive required')
    test_points = sum(g['points'] for g in coverage if g['role']=='test')
    val_points = sum(g['points'] for g in coverage if g['role']=='validation')
    if source['test_points']!=test_points or source['validation_points']!=val_points or source['test_entities']!=len(dataset['entities']):
        raise ValueError('Full source entity and native role point counts required')
    metrics = strict_metrics(dataset,payload,source['metrics']['strict'],c['numerical_metric_tolerance'],source['checkpoint_sha256'])
    if {k:sha(p) for k,p in paths.items()} != before: raise ValueError('Immutable complete source changed during audit')
    return dict(passed=True,canonical_id=job['id'],dataset=job['dataset'],seed=job['seed'],model_config=job['model_config'],
        source_result_sha256=before['result'],checkpoint_sha256=before['checkpoint'],scores_sha256=before['scores'],
        prepared_input_sha256=dataset['input_sha256'],all_array_bindings=bindings,native_coverage=coverage,
        full_test_points=test_points,full_validation_points=val_points,positive_test_labels=sum(sum(payload[g['array_prefix']+'_labels']) for g in dataset['entities']),
        fit_budget=budget,source_epoch_audit=source['epoch_audit'],source_loss_history_observed=losses is not None,
        independent_strict_metrics=metrics,source_checkpoint_neural_replay=False,source_historical_resource_guard_certified=False,
        independent_bootstrap=False,independent_VUS=False,independent_affiliation=False,
        new_complete_model_results=0,new_training_seed_slots=0,new_optimizer_updates=0,new_neural_forwards=0,
        full_TAB_equivalence=False,paper_equivalence=False,all_original_experiments_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register',action='store_true'); parser.add_argument('--probe-only',action='store_true'); parser.add_argument('--case')
    args = parser.parse_args()
    if args.register:
        with (ROOT/CONFIG).open('x',encoding='utf-8') as f: json.dump(registration(),f,indent=2)
        return
    c = read(ROOT/CONFIG); q,manifest = verify(c)
    target = Path(c['artifact_root'])
    if args.probe_only:
        print(json.dumps(dict(passed=True,required_CPU_slots=len(q['jobs']),numpy_imported='numpy' in sys.modules,
            torch_imported='torch' in sys.modules,sklearn_imported='sklearn' in sys.modules,numerical_workers_started=0))); return
    if args.case:
        job = next(j for j in q['jobs'] if j['id']==args.case)
        report = audit(c,q,manifest,job)
        path = target/'conditions'/(job['id']+'.json'); path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8') as f: json.dump(report,f,indent=2,allow_nan=False)
        return
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = install_facades(); base = ROOT/c['state_root']; base.mkdir(parents=True,exist_ok=True)
    owner = api['exclusive_lock'](base/'controller.lock')
    if owner is None: raise RuntimeError('Full550 independent metric collector already live')
    if (base/'status.json').exists(): raise RuntimeError('Preserve previous collector state and terminal attempts')
    state = dict(pid=os.getpid(),create_time=psutil.Process().create_time(),required_original_CPU_slots=550,
        verified_ids=[],failed_preserved={},active_pid=None,all_original_experiments_complete=False)
    def update(v):
        state.update(v,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        while len(state['verified_ids'])+len(state['failed_preserved']) < 550:
            for job in q['jobs']:
                identity = job['id']
                if identity in state['verified_ids'] or identity in state['failed_preserved']: continue
                if not (ROOT/job['output_directory']/'result.json').exists(): continue
                with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],update,poll_seconds=10):
                    attempt = base/'attempts'/identity; attempt.mkdir(parents=True,exist_ok=False)
                    command = [sys.executable,'-B','-X','utf8','-u','-m',MODULE,'--case',identity]
                    with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
                        child = subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                        update(dict(state='independent_complete_saved_score_audit',active_pid=child.pid,
                            active_create_time=psutil.Process(child.pid).create_time(),active_condition=identity,actual_command=command))
                        code,usage = capped_wait(child,c,api,update)
                    proof = target/'conditions'/(identity+'.json')
                    passed = code==0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
                    receipt = dict(passed=passed,exit_code=code,**usage,report_sha256=sha(proof) if proof.exists() else None)
                    write(attempt/'receipt.json',receipt)
                    if passed: state['verified_ids'].append(identity)
                    else: state['failed_preserved'][identity] = receipt
                    update(dict(active_pid=None,state='waiting_complete_original_source_results'))
            update(dict(state='waiting_complete_original_source_results',active_pid=None));time.sleep(10)
        update(dict(state='all_source_metric_attempts_terminal',active_pid=None))
    finally: owner.close()


if __name__=='__main__': main()
