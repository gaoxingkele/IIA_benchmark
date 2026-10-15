"""Full-source analytic original-random-mask compatibility; no model execution.

Each case is checked; repeated channel groups and fixed masks across stride-one
windows are counted exactly without replaying the same 256-bit mask millions
of times. No raw case, window, channel group or zero-target item is dropped.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime,timezone
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import time
from scripts.flow_matching import bind_moment_icm_full_training_inputs_v1 as inputs
from scripts.flow_matching import moment_icm_pile_cache_reader_v1 as data
from scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1 import status_writer

ROOT,read,sha,write=inputs.ROOT,inputs.read,inputs.sha,inputs.write
MODULE='scripts.flow_matching.audit_moment_icm_full_training_masks_v1'
CONFIG='configs/reproducibility/fm_moment_ICM_all2086_full_training_masks.2026-10-15_v1.json'
REPORT='all2086_training_mask_compatibility.json'
TRAIN_BODY='configs/reproducibility/fm_moment_ICM_valid_target_training_body.2026-10-15_v1.json'


def ready_binding():
    config=read(ROOT/inputs.CONFIG)
    report=Path(config['artifact_root'])/inputs.REPORT
    guard=ROOT/config['state_root']/'guard_receipt.json'
    if not report.exists() or not guard.exists():return None
    proof,g=read(report),read(guard)
    if (not proof['passed'] or proof['configuration_sha256']!=sha(ROOT/inputs.CONFIG)
            or proof['source_count']!=2086 or not proof['complete_conditional_recipe_model_inputs_bound']
            or not g['passed'] or g['exit_code']!=0 or g['resource_guard_aborted'] or g['proof_sha256']!=sha(report)):
        raise ValueError('Actual complete successful2086 input binding and guard required')
    return proof


def analyze_padding(mask):
    import numpy as np
    mask=np.asarray(mask)
    if mask.shape!=(256,) or not np.isin(mask,[0,1]).all():raise ValueError('Exact binary source256 mask required')
    patches=int((mask.reshape(32,8).sum(-1)==8).sum())
    # Original Masking: noise on observed patches is in [0,1); padding noise=1.
    # argsort chooses ceil(0.7*N) observed patches. Thus this guarantee is exact
    # for every possible source random draw, without consuming Torch RNG.
    keep=math.ceil(patches*.7)
    return dict(observed_time_points=int(mask.sum()),complete_observed_patches=patches,
        retained_patches_for_every_source_random_draw=keep,guaranteed_retained_observations=keep>0)


def inspect_source(source,record,arrays):
    directory=Path(source['directory'])/('TRAIN' if source['family']=='classification' else '')
    family=source['family'];casewise=family in ('classification','monash')
    mask_path=directory/('input_mask.npy' if casewise else 'input_mask_window.npy')
    pins={str(Path(path)):digest for path,digest in record['input_files_sha256'].items()}
    if str(mask_path) not in pins or sha(mask_path)!=pins[str(mask_path)]:raise ValueError('Complete frozen source padding mask required')
    shape=arrays.header(mask_path)['shape']
    cases=source['training_base_cases_or_windows'];groups=source['channel_groups']
    if source['items']!=cases*groups:raise ValueError('All original cases/windows and channel groups required')
    bad=[];histogram={};observed=0;minimum=32
    if casewise:
        if len(shape)!=2 or shape[1]!=256:raise ValueError('Complete original case/time mask axes required')
        if family=='monash':
            start,end=source['source_train_shuffle_positions']
            if end-start!=cases:raise ValueError('Complete original Monash training shuffle positions required')
            shuffle=directory/'source_shuffle_order.npy'
            if str(shuffle) not in pins or sha(shuffle)!=pins[str(shuffle)]:raise ValueError('Original shuffled case identities required')
        for base in range(cases):
            case=(int(arrays.flat(shuffle,start+base,1)[0]) if family=='monash' else base)
            if not 0<=case<shape[0]:raise ValueError('Original cache case row required')
            result=analyze_padding(arrays.flat(mask_path,case*256,256))
            n=result['complete_observed_patches'];histogram[str(n)]=histogram.get(str(n),0)+groups
            observed+=result['observed_time_points']*groups*source['width'];minimum=min(minimum,n)
            if not result['guaranteed_retained_observations']:bad.append(dict(training_base_case=base,cache_case=case,affected_items=groups))
    else:
        if shape!=(256,):raise ValueError('Complete shared source time mask required')
        result=analyze_padding(arrays.flat(mask_path,0,256));minimum=result['complete_observed_patches']
        histogram[str(minimum)]=source['items'];observed=result['observed_time_points']*source['items']*source['width']
        if not result['guaranteed_retained_observations']:bad.append(dict(all_stride1_base_windows=cases,affected_items=source['items']))
    if sum(histogram.values())!=source['items']:raise ValueError('Every training item must be covered by this analytic mask audit')
    return dict(dataset_key=source['dataset_key'],family=family,source_items=source['items'],base_cases_or_windows=cases,
        distinct_cached_case_masks_inspected=cases if casewise else 1,
        identical_stride1_window_mask_reused_analytically=not casewise,
        full_patch_count_histogram_over_all_training_items=histogram,minimum_complete_observed_patches=minimum,
        input_observed_tensor_coordinates_including_channel_groups=observed,
        incompatible_cases_or_shared_masks=bad,incompatible_training_items=sum(r['affected_items'] for r in bad),
        source_mask_sha256=pins[str(mask_path)],guaranteed_nonempty_RevIN_keep_for_every_original_random_draw=not bad,
        raw_target_validity_or_neural_performance_certified=False)


def registration():
    previous=read(ROOT/inputs.CONFIG);inputs.verify(previous)
    body=read(ROOT/TRAIN_BODY)
    files=[inputs.CONFIG,TRAIN_BODY,MODULE.replace('.','/')+'.py','tests/test_fm_moment_icm_full_training_masks_v1.py']
    files += [p['path'] for p in previous['source_receipts']]+[p['path'] for p in body['source_receipts']]
    return dict(schema_version=1,paper_id='moment_long_context',paper_citation=body['paper_citation'],data_citation=body['data_citation'],
        source_references=2086,epoch_items=80850189,batch_size=2048,epochs=1,mask_ratio=.3,patch_len=8,seq_len=256,
        original_mask_sha256=body['original_numerical_source_sha256']['moment/utils/masking.py'],
        environment=previous['environment'],python=previous['python'],settings=previous['settings'],resource_lock=previous['resource_lock'],
        cache_bytes=64*2**20,block_bytes=2**20,source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(set(files))],
        state_root='experiments/runs/fm_moment_icm_all2086_full_training_masks_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_all2086_full_training_masks_v1',
        boundary=('All2086sources and80,850,189trainingitems;full eachcase padding masks including Monash '
            'original trainshuffle positions,allchannelgroups andall stride1windows of sharedconstantmask. '
            'Exact original keepceil(0.7*observedfullpatches) analytic guarantee foranyrandomdraw; noTorchRNG '
            'consumed. No rowdrop,implicitfallback,mask/preprocess/data/modelchange or performance. Require '
            'actual full2086 binding+322independentclosure+fullindexproof first; original16/21/16/3GiB '
            'CPUdata policy/lock unchanged. Raw-target validity stays separate frominputmask compatibility.'))


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen full mask audit source changed: '+pin['path'])
    original,sources=inputs.verify(read(ROOT/inputs.CONFIG))
    previous=read(ROOT/inputs.CONFIG)
    if (config['source_references']!=2086 or config['epoch_items']!=80850189 or config['batch_size']!=2048 or config['epochs']!=1
            or config['mask_ratio']!=.3 or config['patch_len']!=8 or config['seq_len']!=256
            or config['cache_bytes']!=64*2**20 or config['block_bytes']!=2**20
            or any(config[key]!=previous[key] for key in ['settings','resource_lock','python','environment'])
            or config['original_mask_sha256']!=read(ROOT/TRAIN_BODY)['original_numerical_source_sha256']['moment/utils/masking.py']):
        raise ValueError('Full original source/item/mask recipe and unchanged resource policy required')
    return sources


def worker(config):
    sources=verify(config);binding=ready_binding()
    if binding is None:raise ValueError('Actual complete input binding must finish first')
    if binding['sources']!=sources:raise ValueError('Identical complete2086source descriptors required')
    actual=dict(python=platform.python_version(),**{key:importlib.metadata.version(key) for key in ['numpy','pandas','scipy','scikit-learn']})
    if actual!=config['environment']:raise ValueError('Exact mask audit environment required')
    records={r['dataset_key']:r for r in binding['source_records']}
    if len(records)!=2086 or set(records)!={s['dataset_key'] for s in sources}:raise ValueError('Every source binding required')
    output=Path(config['artifact_root']);output.mkdir(parents=True,exist_ok=False)
    progress=status_writer(output/'progress.json',write);rows=[]
    arrays=data.BoundedArrays(cache_bytes=config['cache_bytes'],block_bytes=config['block_bytes'])
    try:
        for s in sources:
            rows.append(inspect_source(s,records[s['dataset_key']],arrays))
            progress(output/'progress.json',dict(audited_sources=len(rows),required=2086))
    finally:arrays.close()
    if len(rows)!=2086 or sum(r['source_items'] for r in rows)!=80850189:raise ValueError('Full2086source/80mitemscope required')
    bad=sum(r['incompatible_training_items'] for r in rows)
    write(output/REPORT,dict(passed=bad==0,captured_utc=datetime.now(timezone.utc).isoformat(),configuration=CONFIG,
        configuration_sha256=sha(ROOT/CONFIG),sources=2086,epoch_items=80850189,records=rows,
        incompatible_training_items=bad,every_case_and_shared_window_mask_analytically_covered=True,
        every_original_random_mask_draw_has_nonempty_RevIN_inputs=bad==0,
        peak_cached_mapping_bytes=arrays.peak_mapped_bytes,neural_training_or_Torch_RNG_draws=0,
        raw_targets_or_model_performance_certified=False,all_original_experiments_complete=False))


def controller(config):
    import msvcrt,psutil
    base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    lock=(base/'prerequisite_metadata.lock').open('a+b')
    if lock.tell()==0:lock.write(b'0');lock.flush()
    lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:lock.close();raise RuntimeError('Existing full source mask audit owner preserved')
    publish=status_writer(base/'status.json',write)
    try:
        while ready_binding() is None:
            publish(base/'status.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,
                state='waiting_actual_all2086_training_input_binding',required_sources=2086))
            upstream=ROOT/read(ROOT/inputs.CONFIG)['state_root']/'status.json'
            if upstream.exists() and read(upstream).get('state')=='failed_preserved_without_retry':
                raise ValueError('Full input binding failed; mask audit not launched')
            time.sleep(10)
        path=ROOT/'scripts/flow_matching/verify_moment_icm_PILE_anomaly_v1.py'
        node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='controller')
        labels={'closing_all1949_anomaly_sources':'auditing_all2086_full_training_masks',
            'all1949_anomaly_cache_closure_passed':'all2086_full_training_masks_compatible'}
        class Labels(ast.NodeTransformer):
            def visit_Constant(self,node):
                if isinstance(node.value,str) and node.value in labels:return ast.copy_location(ast.Constant(labels[node.value]),node)
                return node
        node=Labels().visit(node);ast.fix_missing_locations(node)
        namespace=dict(inputs.epoch.template.controller.__globals__);namespace.update(MODULE=MODULE,CONFIG=CONFIG,REPORT=REPORT)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<full-mask-original-data-resource-guard>','exec'),namespace)
        namespace['controller'](config)
    finally:lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true')
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.register:
        if (ROOT/CONFIG).exists():raise FileExistsError('Preserve frozen full mask audit registration')
        config=registration();verify(config)
        with (ROOT/CONFIG).open('x',encoding='utf-8',newline='\n') as stream:
            __import__('json').dump(config,stream,indent=2);stream.write('\n')
    else:config=read(ROOT/CONFIG);verify(config)
    if args.probe_only:print(__import__('json').dumps(dict(passed=True,sources=2086,epoch_items=80850189,Torch_RNG_draws=0)))
    elif args.worker:worker(config)
    else:controller(config)


if __name__=='__main__':main()
