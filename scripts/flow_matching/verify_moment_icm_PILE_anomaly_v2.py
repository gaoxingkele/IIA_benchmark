"""Full1949 cache closure with exact original-success/diagnosed-repair partition."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import platform
from scripts.flow_matching import verify_moment_icm_PILE_anomaly_v1 as checks
from scripts.flow_matching import recover_moment_icm_anomaly_scipy1101_v2 as recovery
from scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1 import status_writer

ROOT,read,sha,write=checks.ROOT,checks.read,checks.sha,checks.write
MODULE='scripts.flow_matching.verify_moment_icm_PILE_anomaly_v2'
CONFIG='configs/reproducibility/fm_moment_ICM_PILE_all1949_anomaly_closure.2026-10-15_v2.json'
REPORT='full1949_anomaly_cache_closure_v2.json'


def choose_records(original_jobs,original_success,repaired_records):
    expected={j['id'] for j in original_jobs}
    if (len(expected)!=1949 or set(original_success)&set(repaired_records)
            or set(original_success)|set(repaired_records)!=expected):
        raise ValueError('Exactly1949original sources with disjoint success/diagnosed-repair ownership required')
    return [(j,'original' if j['id'] in original_success else 'dependency_repair') for j in original_jobs]


def registration():
    previous=read(ROOT/checks.producer.CONFIG);repair=read(ROOT/recovery.CONFIG)
    pins=[checks.producer.CONFIG,recovery.CONFIG,checks.CONFIG,
        'scripts/flow_matching/verify_moment_icm_PILE_anomaly_v1.py',
        'scripts/flow_matching/verify_moment_icm_PILE_anomaly_v2.py',
        'tests/test_fm_moment_icm_PILE_anomaly_closure_v2.py']
    pins += [p['path'] for p in read(ROOT/checks.CONFIG)['source_receipts']]
    return dict(schema_version=2,paper_id='moment_long_context',paper_citation=previous['paper_citation'],
        producer_configuration=checks.producer.CONFIG,recovery_configuration=recovery.CONFIG,
        sources=1949,families=18,cells=5847,arrays=17541,
        original_successful_sources=repair['original_successful_sources'],
        diagnosed_dependency_repair_sources=repair['required_failed_source_repairs'],
        environment=previous['environment'],python=previous['python'],settings=previous['settings'],resource_lock=previous['resource_lock'],
        state_root='experiments/runs/fm_moment_icm_PILE_all1949_anomaly_closure_v2',
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_PILE_all1949_anomaly_closure_v2',
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(set(pins))],
        boundary=('All1949source references kept. Original successful caches and separate failed-only '
            'SciPy1.10.1 compatibleenvironment reconstruction remain explicit disjoint inputs. '
            'Preserve322originaldependency failures,no original/successful cache replacement/retry. '
            'Rehash every raw/cache/proof/guard,independently derive CSVrecord/header/pair/split/ '
            'fitextent/gate,check every finite signal/rowID/mask. Noindependentfullnormalization '
            'rewrite or historicalICM/TAB/numericalequivalence certification across environments. '
            'No neuralfit/inference/SVM,unchanged16/21/16/3GiBdata policy.'))


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen mixed-cache closure input changed: '+pin['path'])
    original=read(ROOT/config['producer_configuration']);jobs=checks.producer.verify(original)
    repair=read(ROOT/config['recovery_configuration']);failed=recovery.verify(repair)
    if (config['sources']!=1949 or config['families']!=18 or config['cells']!=5847 or config['arrays']!=17541
            or config['original_successful_sources']+config['diagnosed_dependency_repair_sources']!=1949
            or config['diagnosed_dependency_repair_sources']!=len(failed)
            or config['settings']!=original['settings'] or config['environment']!=original['environment']):
        raise ValueError('Exact full sources/partition and unchanged independent data audit environment required')
    return original,jobs,repair,failed


def worker(config):
    actual=dict(python=platform.python_version(),**{k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn']})
    if actual!=config['environment']:raise ValueError('Exact independent original data environment required')
    original,jobs,repair,failed=verify(config)
    original_ok,failed_again,original_pins=recovery.source_partition(original,jobs)
    if failed_again!=failed:raise ValueError('Original failure partition changed')
    state=read(ROOT/repair['state_root']/'status.json')
    families=sorted({j['family'] for j in failed})
    if state['state']!='all_diagnosed_repair_family_attempts_terminal' or sorted(state['completed_families'])!=families or state['failed_preserved']:
        raise ValueError('All diagnosed compatibleenvironment preparations must succeed first')
    repaired={};repair_chain=[]
    for family in families:
        path=Path(repair['artifact_root'])/family/'family_preparation.json';proof=read(path)
        gp=ROOT/repair['state_root']/'attempts'/family/'guard_receipt.json';guard=read(gp)
        subset=[j for j in failed if j['family']==family]
        if (not proof['passed'] or proof['failed_preserved'] or proof['configuration_sha256']!=sha(ROOT/recovery.CONFIG)
                or proof['environment']!=repair['environment'] or not guard['passed'] or guard['exit_code']!=0
                or guard['resource_guard_aborted'] or guard['proof_sha256']!=sha(path)
                or guard['exact_child_handle']['command']!=[repair['python'],'-B','-X','utf8','-u','-m',recovery.MODULE,'--family',family]
                or [r['id'] for r in proof['records']]!=[j['id'] for j in subset]):
            raise ValueError('Complete distinct compatibleenvironment source and guard chain required')
        repaired.update({r['id']:r for r in proof['records']});repair_chain.append(dict(proof=str(path),proof_sha256=sha(path),guard=str(gp),guard_sha256=sha(gp)))
    selected=choose_records(jobs,original_ok,repaired)
    output=Path(config['artifact_root']);output.mkdir(parents=True,exist_ok=False)
    publish=status_writer(output/'progress.json',write);results=[]
    for job,track in selected:
        source_config=original if track=='original' else repair
        source_config_path=checks.producer.CONFIG if track=='original' else recovery.CONFIG
        source_record=(original_ok if track=='original' else repaired)[job['id']]
        directory=Path(source_config['artifact_root'])/job['family']/job['id']
        proof_path=directory/'pretraining_preparation.json';proof=read(proof_path)
        if (not proof['passed'] or proof['job']!=job or sha(proof_path)!=source_record['sha256']
                or proof['configuration_sha256']!=sha(ROOT/source_config_path)
                or not proof['original_complete_class_AST_reused'] or not proof['original_header_split_scaler_unmodified']
                or proof['neural_fit_inference_SVM']!=0 or proof['historical_ICM_pipeline_certified']):
            raise ValueError('Exact selected original source method and every source proof required')
        rows,raw=checks.raw_records(job['raw']);train_rows=train_raw=None
        if job['train_raw']:train_rows,train_raw=checks.raw_records(job['train_raw'])
        for audit,record in [(raw,proof['raw_CSV_audit']),(train_raw,proof['paired_NASA_train_audit'])]:
            if audit is None:
                if record is not None:raise ValueError('Unexpected nonNASA pair')
            elif (audit['first_record']!=record['first_CSV_record'] or audit['parsed_source_rows']!=record['parsed_rows_with_original_header_infer']
                    or audit['first_record_numeric_finite']!=record['first_record_numeric_finite'] or audit['sha256']!=record['sha256']):
                raise ValueError('Independent original raw header/count/identity differs')
        split_rows,fit=checks.expected_splits(job,rows,train_rows);splits={}
        for split in checks.producer.SPLITS:
            raw_path=job['train_raw']['path'] if job['train_raw'] and split!='test' else job['raw']['path']
            splits[split]=checks.close_split(directory/split,proof['splits'][split],split_rows[split],fit,job['family'],raw_path)
        results.append(dict(id=job['id'],family=job['family'],track=track,source_environment=source_config['environment'],
            raw=raw,NASA_paired_train_raw=train_raw,cache_directory=str(directory),source_proof_sha256=sha(proof_path),splits=splits))
        publish(output/'progress.json',dict(completed_sources=len(results),required_sources=1949))
    write(output/REPORT,dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),configuration=CONFIG,configuration_sha256=sha(ROOT/CONFIG),
        sources=1949,families=18,cells=5847,arrays=17541,original_successful_sources=len(original_ok),
        diagnosed_environment_recovery_sources=len(repaired),original_dependency_failures_preserved=True,
        full_signal_coordinates=sum(c['signal_coordinates'] for r in results for c in r['splits'].values()),
        source_training_gate_selected=sum(r['splits']['train']['source_length_gate_selected'] for r in results),
        source_training_windows=sum(r['splits']['train']['eligible_windows'] for r in results),
        numeric_first_record_omitted_sources=sum(r['raw']['first_record_numeric_finite'] for r in results),
        NASA_pair_references=sum(r['NASA_paired_train_raw'] is not None for r in results),records=results,
        original_family_proof_chain=original_pins,repair_family_proof_chain=repair_chain,
        independent_environment=actual,independent_normalization_reconstructed=False,
        strict_temporal_independence_certified=False,historical_ICM_pipeline_certified=False,
        neural_fit_inference_SVM=0,all_original_experiments_complete=False))


def controller(config):
    path=ROOT/'scripts/flow_matching/verify_moment_icm_PILE_anomaly_v1.py'
    node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='controller')
    ns=dict(checks.controller.__globals__);ns.update(MODULE=MODULE,CONFIG=CONFIG,REPORT=REPORT)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path)+'::<isolated-guard-controller>','exec'),ns)
    ns['controller'](config)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true')
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.register:
        if (ROOT/CONFIG).exists():raise FileExistsError('Preserve frozen v2 closure')
        config=registration();verify(config)
        with (ROOT/CONFIG).open('x',encoding='utf-8',newline='\n') as stream:json.dump(config,stream,indent=2);stream.write('\n')
    else:config=read(ROOT/CONFIG);verify(config)
    if args.probe_only:print(json.dumps(dict(passed=True,sources=1949,cells=5847,arrays=17541,original_successes=config['original_successful_sources'],failed_only_repairs=config['diagnosed_dependency_repair_sources'])))
    elif args.worker:worker(config)
    else:controller(config)


if __name__=='__main__':main()
