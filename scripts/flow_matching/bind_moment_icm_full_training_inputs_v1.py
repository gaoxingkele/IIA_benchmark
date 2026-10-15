"""Bind every2086 selected source to bounded model inputs and target validity.

Requires actual full322control closure and complete80million index plan first.
No model training, performance or historical paper equivalence is certified.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
from scripts.flow_matching import prepare_moment_icm_full_epoch_plan_v1 as epoch
from scripts.flow_matching import verify_moment_icm_anomaly_finite_control_v1 as finite
from scripts.flow_matching import moment_icm_pile_cache_reader_v1 as reader
from scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1 import status_writer

ROOT,read,sha,write=epoch.ROOT,epoch.read,epoch.sha,epoch.write
MODULE='scripts.flow_matching.bind_moment_icm_full_training_inputs_v1'
CONFIG='configs/reproducibility/fm_moment_ICM_all2086_training_input_bindings.2026-10-15_v1.json'
REPORT='all2086_training_input_bindings.json'


def passed_upstream(configuration,report_name):
    c=read(ROOT/configuration);report=Path(c['artifact_root'])/report_name;guard=ROOT/c['state_root']/'guard_receipt.json'
    if not report.exists() or not guard.exists():return None
    r,g=read(report),read(guard)
    if (not r['passed'] or r['configuration_sha256']!=sha(ROOT/configuration) or not g['passed'] or g['exit_code']!=0
            or g['resource_guard_aborted'] or g['proof_sha256']!=sha(report)):
        raise ValueError('Actual full successful upstream report and guard required: '+configuration)
    return r


def prerequisites(config):
    a=passed_upstream(epoch.CONFIG,epoch.REPORT);b=passed_upstream(finite.CONFIG,finite.REPORT)
    if a is None or b is None:return False
    if (a['index_closure']['complete_epoch_items']!=80850189 or not a['index_closure']['all_original_source_items_exactly_once']
            or b['sources']!=322 or b['arrays']!=6762 or not b['all_model_inputs_finite']):
        raise ValueError('Complete full epoch and all322transformed input controls required')
    return True


def expected_files(source,registries,finite_records):
    directory=Path(source['directory']);family=source['family'];name=source['dataset_key'].split(':',1)[1]
    if family=='classification':
        e=registries['classification'][name];directory=directory/'TRAIN'
        return {str(directory/file):digest for file,digest in e['cache_files_sha256'].items()}
    if family=='long_forecast':
        e=registries['long_forecast'][name];mapping=dict(values='values_TC.npy',original_rows='original_CSV_rows.npy',window_mask='input_mask_window.npy')
        return {str(directory/mapping[key]):record['sha256'] for key,record in e['files'].items()}
    if family=='monash':
        return {str(directory/file):digest for file,digest in registries['monash'][name]['files_sha256'].items()}
    if family=='anomaly_original':
        e=registries['anomaly_original'][name];mapping=dict(values='values_T.npy',rows='raw_CSV_record_indices.npy',mask='input_mask_window.npy')
        return {str(directory/mapping[key]):record['sha256'] for key,record in e['splits']['train']['files'].items()}
    if family=='anomaly_finite_control':
        e=finite_records[name]
        if str(directory.parent)!=e['cache_directory']:raise ValueError('Exact repaired cache directory required')
        names=finite.NAMES
        return {str(directory/names[key]):digest for key,digest in e['splits']['train']['files_sha256'].items()}
    raise ValueError('Every registered source family must be bound')


def registration():
    original=read(ROOT/epoch.CONFIG);epoch.verify(original)
    files=[epoch.CONFIG,finite.CONFIG,epoch.CLASS,epoch.LONG,epoch.MONASH,epoch.ANOMALY,
        'scripts/flow_matching/moment_icm_pile_cache_reader_v1.py',
        'scripts/flow_matching/bind_moment_icm_full_training_inputs_v1.py',
        'tests/test_fm_moment_icm_pile_cache_reader_v1.py','tests/test_fm_moment_icm_full_training_input_bindings_v1.py']
    files += [p['path'] for p in read(ROOT/finite.CONFIG)['source_receipts']]
    return dict(schema_version=1,paper_id='moment_long_context',paper_citation=original['paper_citation'],data_citation=original['data_citation'],
        epoch_configuration=epoch.CONFIG,finite_control_closure_configuration=finite.CONFIG,expected_sources=2086,
        original_anomaly_sources_requiring_raw_target_audit=1627,finite_control_sources=322,
        header_and_mapping_cache_bytes=64*2**20,mapping_block_bytes=2**20,
        environment=original['environment'],python=original['python'],settings=original['settings'],resource_lock=original['resource_lock'],
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(set(files))],
        state_root='experiments/runs/fm_moment_icm_all2086_training_input_bindings_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_all2086_training_input_bindings_v1',
        boundary=('All2086eligible sources;no missing322silentlyomitted. Require actual full80mepochreference '
            'bijection+322finitecontrolclosure. Rehash allsourcecachefiles,build rawsignal observationtarget '
            'bindings forall1627originalanomaly sources;neverinferobservedfrominterpolatedfiniteinputs. '
            'Sourceclass/Monash paddingtargetsemantics explicitly retained,not rawinternalmissingness '
            'certification. Bounded readerfirst/middle/last example geometry per2086sourcechecked; '
            'not80mcomplete tensor replay or neuraltraining. Keep all sourcevalues/dtypes/rows, '
            'rawdatasetkeyforchannelremainderRNG,MonashshuffleTRAINpositions,TC/CTaxes. '
            'Same16/21/16/3GiBpolicy;no labels/neuralfit,raw/oldcache overwrite orhistorical/TABcausal '
            'equivalence. Fullneuralrunner stillmustapplyvalidtargetswithrandommaskandpadding.'))


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen full training binding source changed: '+pin['path'])
    original=read(ROOT/config['epoch_configuration']);sources=epoch.verify(original)
    if (config['expected_sources']!=2086 or len(sources)!=2086 or config['original_anomaly_sources_requiring_raw_target_audit']!=1627
            or config['finite_control_sources']!=322 or config['settings']!=original['settings']
            or config['environment']!=original['environment'] or config['header_and_mapping_cache_bytes']!=64*2**20
            or config['mapping_block_bytes']!=2**20 or config['python']!=original['python']
            or config['resource_lock']!=original['resource_lock'] or config['epoch_configuration']!=epoch.CONFIG
            or config['finite_control_closure_configuration']!=finite.CONFIG):
        raise ValueError('Full2086source scope and fixed bounded data policy required')
    return original,sources


def worker(config):
    original,sources=verify(config)
    if not prerequisites(config):raise ValueError('All full-source prerequisites must succeed first')
    actual=dict(python=platform.python_version(),**{k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn']})
    if actual!=config['environment']:raise ValueError('Exact source binding environment required')
    registries={
        'classification':{e['dataset']:e for e in read(ROOT/epoch.CLASS)['entries']},
        'long_forecast':{e['dataset']:e for e in read(ROOT/epoch.LONG)['entries'] if e['track']=='paired_full_multivariate' and e['split']=='train'},
        'monash':{e['dataset']:e for e in read(ROOT/epoch.MONASH)['entries']},
        'anomaly_original':{e['id']:e for e in read(ROOT/epoch.ANOMALY)['entries']}}
    f=passed_upstream(finite.CONFIG,finite.REPORT);finite_records={e['id']:e for e in f['records']}
    output=Path(config['artifact_root']);output.mkdir(parents=True,exist_ok=False)
    progress=status_writer(output/'progress.json',write);policies={};records=[]
    r=reader.PileCacheReader(sources,epoch_seed=original['epoch_seed'],original_anomaly_target_bindings=policies,
        cache_bytes=config['header_and_mapping_cache_bytes'],block_bytes=config['mapping_block_bytes'])
    try:
        for index,source in enumerate(sources):
            files=expected_files(source,registries,finite_records)
            for path,digest in files.items():
                if sha(path)!=digest:raise ValueError('Every frozen model input/cache file hash must match: '+path)
            if source['family']=='anomaly_original':
                key=source['dataset_key'];policy=reader.audit_original_anomaly_targets(source,ROOT,output/'raw_target_bindings'/key.split(':',1)[1])
                policies[key]=policy
            checked=[]
            for sample in sorted({0,source['items']//2,source['items']-1}):
                item=r.item(index,sample)
                checked.append(dict(item=sample,shape=list(item['values'].shape),dtype=str(item['values'].dtype),
                    value_bytes_sha256=__import__('hashlib').sha256(item['values'].tobytes()).hexdigest(),
                    valid_targets=int(item['valid_targets'].sum()),target_validity_rule=item['provenance']['target_validity_rule']))
            records.append(dict(dataset_key=source['dataset_key'],family=source['family'],input_files_sha256=files,
                full_source_input_hashes_checked=True,example_geometry_checks=checked,all_possible_examples_exhaustively_read=False))
            progress(output/'progress.json',dict(bound_sources=len(records),required=2086,raw_anomaly_target_profiles=len(policies)))
    finally:r.close()
    if len(records)!=2086 or len(policies)!=1627:raise ValueError('Every source and original raw-observation profile required')
    write(output/REPORT,dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),configuration=CONFIG,
        configuration_sha256=sha(ROOT/CONFIG),sources=sources,source_records=records,original_anomaly_target_bindings=policies,
        source_count=2086,complete_conditional_recipe_model_inputs_bound=True,epoch_reference_configuration=epoch.CONFIG,
        peak_cached_mapping_bytes=r.arrays.peak_mapped_bytes,cache_capacity_bytes=config['header_and_mapping_cache_bytes'],
        original_dataset_values_and_rows_unchanged=True,model_train_or_performance_executed=False,
        full80m_example_materialization_exhaustively_replayed=False,full_neural_training_runner_integration=False,
        historical_pipeline_equivalence_certified=False,all_original_experiments_complete=False))


def controller(config):
    import msvcrt,psutil
    base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    lock=(base/'prerequisite_metadata.lock').open('a+b')
    if lock.tell()==0:lock.write(b'0');lock.flush()
    lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:lock.close();raise RuntimeError('Existing full input binding owner preserved')
    publish=status_writer(base/'status.json',write)
    try:
        while not prerequisites(config):
            publish(base/'status.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,
                state='waiting_full322_control_closure_and_full_epoch_reference_plan',required_sources=2086))
            time.sleep(10)
        path=ROOT/'scripts/flow_matching/verify_moment_icm_PILE_anomaly_v1.py'
        node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='controller')
        replacements={'closing_all1949_anomaly_sources':'binding_all2086_training_cache_sources',
            'all1949_anomaly_cache_closure_passed':'all2086_training_cache_sources_bound'}
        class Names(ast.NodeTransformer):
            def visit_Constant(self,n):
                if isinstance(n.value,str) and n.value in replacements:return ast.copy_location(ast.Constant(replacements[n.value]),n)
                return n
        node=Names().visit(node);ast.fix_missing_locations(node)
        ns=dict(epoch.template.controller.__globals__);ns.update(MODULE=MODULE,CONFIG=CONFIG,REPORT=REPORT)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path)+'::<full-cache-binding-guard>','exec'),ns)
        ns['controller'](config)
    finally:lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true')
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.register:
        if (ROOT/CONFIG).exists():raise FileExistsError('Preserve frozen full training source binding')
        config=registration();verify(config)
        with (ROOT/CONFIG).open('x',encoding='utf-8',newline='\n') as stream:json.dump(config,stream,indent=2);stream.write('\n')
    else:config=read(ROOT/CONFIG);verify(config)
    if args.probe_only:print(json.dumps(dict(passed=True,sources=2086,original_raw_target_profiles=1627,finite_controls=322)))
    elif args.worker:worker(config)
    else:controller(config)


if __name__=='__main__':main()
