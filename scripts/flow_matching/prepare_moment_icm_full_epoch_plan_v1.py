"""Full PILE paired one-epoch reference plan; no data readiness or model claim.

Retain every candidate and explicit original eligibility choice. All eligible
sources are indexed even when322 finite-control caches are still pending.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from scripts.flow_matching import moment_icm_disk_epoch_plan_v1 as sampler
from scripts.flow_matching import verify_moment_icm_PILE_anomaly_v1 as template
from scripts.flow_matching import prepare_moment_icm_anomaly_finite_control_v3 as finite
from scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1 import status_writer

ROOT,read,sha,write=template.ROOT,template.read,template.sha,template.write
MODULE='scripts.flow_matching.prepare_moment_icm_full_epoch_plan_v1'
CONFIG='configs/reproducibility/fm_moment_ICM_full_PILE_paired_epoch_plan.2026-10-15_v1.json'
REPORT='full_PILE_epoch_plan_proof.json'
CLASS='configs/reproducibility/fm_moment_ICM_classification_pretraining_cache_index.2026-10-15_v1.json'
LONG='configs/reproducibility/fm_moment_ICM_long_forecast_pretraining_cache_index.2026-10-15_v1.json'
MONASH='configs/reproducibility/fm_moment_ICM_Monash_pretraining_cache_index.2026-10-15_v1.json'
ANOMALY='configs/reproducibility/fm_moment_ICM_anomaly_original1627_closed_cache_index.2026-10-15_v2.json'


def grouped(channels):return (1,1) if channels==1 else (8,(channels+7)//8)


def complete_sources():
    sources=[];excluded=[];c=read(ROOT/CLASS);l=read(ROOT/LONG);m=read(ROOT/MONASH);a=read(ROOT/ANOMALY)
    for e in c['entries']:
        key='classification:'+e['dataset']
        if not e['selected_by_original_raw_length_gate']:
            excluded.append(dict(dataset_key=key,family='classification',reason='original_raw_length_below256'));continue
        width,groups=grouped(e['channels']);cases=e['disclosed_paired_casewise_split']['grouped_casewise_training_cases']
        sources.append(dict(dataset_key=key,family='classification',split='train',width=width,items=cases*groups,
            native_channels=e['channels'],channel_groups=groups,training_base_cases_or_windows=cases,
            directory=e['source_directory'],source_proof_sha256=e['source_proof_sha256'],
            sample_identity='divmod(item,channel_groups)->original_TRAIN_case_prefix,group',
            source_order='casewise6of7_prefix_not_original_flattened_channel_boundary',cache_independent_closure_complete=True))
    for e in l['entries']:
        if e['track']!='paired_full_multivariate' or e['split']!='train':continue
        width,groups=grouped(e['channels']);n=e['complete_window_count']
        sources.append(dict(dataset_key='long_forecast:'+e['dataset'],family='long_forecast',split='train',width=width,items=n*groups,
            native_channels=e['channels'],channel_groups=groups,training_base_cases_or_windows=n,
            directory=e['directory'],source_proof_sha256=e['source_proof_sha256'],
            sample_identity='divmod(item,channel_groups)->stride1_window_start,group',
            cache_independent_closure_complete=True,original_OT_only_source_track_not_pooled=True))
    for e in m['entries']:
        key='monash:'+e['dataset']
        if not e['source_count_gate_selected']:
            excluded.append(dict(dataset_key=key,family='monash',reason='original_source_count_field_gate_not_selected'));continue
        start,end=e['source_split_positions']['train']
        if end-start!=e['source_training_rows']:raise ValueError('Exact source shuffle/train positions required')
        sources.append(dict(dataset_key=key,family='monash',split='train',width=1,items=end-start,native_channels=1,
            channel_groups=1,training_base_cases_or_windows=end-start,directory=e['directory'],
            source_proof_sha256=e['source_proof_sha256'],source_train_shuffle_positions=[start,end],
            sample_identity='source_shuffle_order[train_start+item] indexes original raw-order cache',
            cache_independent_closure_complete=True,source_count_gate_not_temporal_control=True))
    for e in a['entries']:
        s=e['splits']['train'];n=s['complete_stride1_windows']
        if not s['source_length_gate_selected']:raise ValueError('All1627known source training caches must retain original gate')
        sources.append(dict(dataset_key='anomaly:'+e['id'],family='anomaly_original',split='train',width=1,items=n,native_channels=1,
            channel_groups=1,training_base_cases_or_windows=n,directory=str(Path(e['directory'])/'train'),
            source_proof_sha256=e['source_proof_sha256'],sample_identity='original_stride1_window_start',
            cache_independent_closure_complete=True,target_validity='source_finite_mask_and_padding;verify_raw_observation_contract_before_neural_training'))
    fc=read(ROOT/finite.CONFIG);audit_config=read(ROOT/finite.audit.CONFIG)
    proof_path=Path(audit_config['artifact_root'])/finite.audit.REPORT;proof=read(proof_path)
    if not proof['passed'] or proof['source_audits']!=322 or proof['original_successful_caches_independently_closed']!=1627:
        raise ValueError('Actual322source audit plus1627closed originals required')
    by_id={r['id']:r for r in proof['records']};audit_sha=sha(proof_path)
    for job in fc['jobs']:
        row=by_id[job['id']];n=row['splits']['train']['time_rows']-255
        if n<=0:raise ValueError('All322known control sources must preserve exact positive source length gate')
        sources.append(dict(dataset_key='anomaly:'+job['id'],family='anomaly_finite_control',split='train',width=1,items=n,
            native_channels=1,channel_groups=1,training_base_cases_or_windows=n,directory=str(Path(fc['artifact_root'])/job['id']/'train'),
            source_missingness_audit_sha256=audit_sha,sample_identity='restored_source_stride1_window_start',
            cache_independent_closure_complete=False,target_validity='raw_observed AND source_finite:required in random reconstruction loss',
            full322_control_preparation_and_closure_required_before_model_read=True))
    return sources,excluded


def counts(sources,excluded,batch_size):
    layout=sampler.layout(sources,batch_size);families={}
    for s in sources:
        f=families.setdefault(s['family'],dict(sources=0,items=0,tensor_time_positions=0,patch_token_positions=0))
        f['sources']+=1;f['items']+=s['items'];f['tensor_time_positions']+=s['items']*s['width']*256
        f['patch_token_positions']+=s['items']*s['width']*32
    return dict(candidate_getter_references=2162,selected_source_references=len(sources),excluded_source_references=len(excluded),
        families=families,total_items=layout['total_items'],optimizer_batch_slots_per_full_epoch=layout['total_batches'],
        width_buckets=[dict(width=b['width'],items=b['items'],batches=b['batches'],last_batch_items=(b['items']-1)%batch_size+1) for b in layout['buckets']],
        full_tensor_time_positions_including_overlap_and_repeated_remainder_channels=sum(f['tensor_time_positions'] for f in families.values()),
        full_patch_token_positions_including_padding=sum(f['patch_token_positions'] for f in families.values()),
        active_valid_reconstruction_targets_not_yet_certified=True)


def registration():
    sources,excluded=complete_sources();batch=2048;seed=13
    policy=read(ROOT/'configs/reproducibility/fm_moment_ICM_PILE_all9_long_forecast_pretraining.2026-10-15_v1.json')
    source_root=policy['source_root'];yaml=source_root+'/configs/pretraining/pretrain.yaml'
    pins=[CLASS,LONG,MONASH,ANOMALY,finite.CONFIG,finite.audit.CONFIG,yaml,
        'scripts/flow_matching/moment_icm_disk_epoch_plan_v1.py','tests/test_fm_moment_icm_disk_epoch_plan_v1.py',
        'scripts/flow_matching/prepare_moment_icm_full_epoch_plan_v1.py','tests/test_fm_moment_icm_full_epoch_plan_v1.py',
        'scripts/flow_matching/moment_icm_channel_data_v1.py','scripts/flow_matching/verify_moment_icm_PILE_anomaly_v1.py']
    audit_config=read(ROOT/finite.audit.CONFIG);pins += [str(Path(audit_config['artifact_root'])/finite.audit.REPORT)]
    return dict(schema_version=1,paper_id='moment_long_context',paper_citation=policy['paper_citation'],
        data_citation='https://huggingface.co/datasets/AutonLab/Timeseries-PILE',sources=sources,excluded=excluded,
        epoch_seed=seed,batch_size=batch,epochs=1,drop_last=False,data_source_fraction=1.,
        batch_size_origin=dict(path=yaml,field='train_batch_size',related_source_value=2048,exact_paper_pretraining_batch_size_published=False),
        channel_policy='appendix_D_partition8_remainder_replacement',long_forecast_track='paired_full_multivariate',
        classification_train_selection='disclosed_casewise6of7',monash_selection='original_source_count_field',
        count_audit=counts(sources,excluded,batch),settings=policy['settings'],python=policy['python'],environment=policy['environment'],
        resource_lock=policy['resource_lock'],source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(set(pins))],
        state_root='experiments/runs/fm_moment_icm_full_PILE_epoch_plan_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_full_PILE_epoch_plan_v1',
        boundary=('Full2162candidate references,all2086eligible source references. Keep95classification/9fullmultivariatetrain '
            '/33originalcountgateMonash/1627originalanomaly/322explicitfinitecontrol. None silentlyomitted. '
            'Plan fulloneepoch/allstride1windows/casewise6of7 andpaperAppendixDgroups8,includingreplacementremainder. '
            'No maintextsubsample or temporalMonash control silentlysubstituted. Source pretrainingYAMLbatch2048/seed13 '
            'are disclosed relatedsource choices,not undocumented paper hyperparameter certification. '
            'Exact old RandomState permutation andfinalPythonlistbatchshuffle withdiskint64indices,allitems/batches '
            'bijections verified,partialbatches retained. Metadatareferences only;322cachecompletion,targetvalidity '
            'loader/neuralintegration stillrequired. Same16/21/16/3GiBdataresourcepolicy,no modeltraining/inference '
            'or performance/historicalpipeline/TABcausal equivalence claim.'))


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen full epoch plan source changed: '+pin['path'])
    sources,excluded=complete_sources()
    if (config['sources']!=sources or config['excluded']!=excluded or len(sources)!=2086 or len(excluded)!=76
            or config['count_audit']!=counts(sources,excluded,2048) or config['batch_size']!=2048 or config['epoch_seed']!=13
            or config['epochs']!=1 or config['drop_last'] or config['data_source_fraction']!=1.
            or config['channel_policy']!='appendix_D_partition8_remainder_replacement'
            or config['long_forecast_track']!='paired_full_multivariate'
            or config['classification_train_selection']!='disclosed_casewise6of7'
            or config['monash_selection']!='original_source_count_field'
            or config['settings']!=dict(minimum_available_memory_bytes=16*2**30,minimum_commit_headroom_bytes=21*2**30,
                emergency_commit_headroom_bytes=16*2**30,maximum_worker_rss_bytes=3*2**30,maximum_worker_private_bytes=3*2**30)):
        raise ValueError('Complete2162candidate/2086eligible reference scope and full paired epoch required')
    return sources


def worker(config):
    actual=dict(python=platform.python_version(),**{k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn']})
    if actual!=config['environment']:raise ValueError('Exact metadata generator environment required')
    sources=verify(config);root=Path(config['artifact_root']);root.mkdir(parents=True,exist_ok=False)
    progress=status_writer(root/'progress.json',write)
    plan=sampler.build(root/'paired_epoch',sources,config['batch_size'],config['epoch_seed'],lambda x:progress(root/'progress.json',x))
    reader=sampler.DiskEpochPlan(root/'paired_epoch')
    closure=reader.validate_complete(progress=lambda x:progress(root/'progress.json',x))
    if closure['complete_epoch_items']!=config['count_audit']['total_items']:raise ValueError('No source window or case may be dropped')
    write(root/REPORT,dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),configuration=CONFIG,
        configuration_sha256=sha(ROOT/CONFIG),count_audit=config['count_audit'],index_closure=closure,
        epoch_plan_file=str(root/'paired_epoch/epoch_plan.json'),epoch_plan_sha256=sha(root/'paired_epoch/epoch_plan.json'),
        largest_live_native_permutation_bytes=plan['largest_live_native_permutation_bytes'],
        complete_training_data_ready=False,actual_model_or_data_window_materialization_executed=False,
        all_original_experiments_complete=False))


def controller(config):
    path=ROOT/'scripts/flow_matching/verify_moment_icm_PILE_anomaly_v1.py'
    node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='controller')
    replacements={'closing_all1949_anomaly_sources':'building_complete_full_PILE_epoch_references',
        'all1949_anomaly_cache_closure_passed':'full_PILE_epoch_references_complete_and_verified'}
    class Names(ast.NodeTransformer):
        def visit_Constant(self,n):
            if isinstance(n.value,str) and n.value in replacements:return ast.copy_location(ast.Constant(replacements[n.value]),n)
            return n
    node=Names().visit(node);ast.fix_missing_locations(node)
    ns=dict(template.controller.__globals__);ns.update(MODULE=MODULE,CONFIG=CONFIG,REPORT=REPORT)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path)+'::<full-epoch-metadata-guard>','exec'),ns)
    ns['controller'](config)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true')
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.register:
        if (ROOT/CONFIG).exists():raise FileExistsError('Preserve frozen complete epoch plan')
        config=registration();verify(config)
        with (ROOT/CONFIG).open('x',encoding='utf-8',newline='\n') as stream:json.dump(config,stream,indent=2);stream.write('\n')
    else:config=read(ROOT/CONFIG);verify(config)
    if args.probe_only:print(json.dumps(dict(passed=True,**config['count_audit'],torch_imported='torch' in sys.modules)))
    elif args.worker:worker(config)
    else:controller(config)


if __name__=='__main__':main()
