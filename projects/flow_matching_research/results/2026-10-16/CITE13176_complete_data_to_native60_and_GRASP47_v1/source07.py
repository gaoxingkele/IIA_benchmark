"""Wait for the full raw-backed CITE preparation, then run all60 native cases."""
import argparse
import ast
from pathlib import Path
import os
import subprocess
import sys
import time
from types import FunctionType
from datetime import datetime,timezone
from scripts.flow_matching import run_otcfm_EB_CITE_native150_v2 as base
from scripts.flow_matching import prepare_CITE13176_GEO_raw_backed_QC_v2 as preparation

ROOT,read,sha,write,exclusive=base.ROOT,base.read,base.sha,base.write,base.exclusive
MODULE='scripts.flow_matching.run_otcfm_CITE13176_GEO_native60_v1'
QUEUE='configs/experiments/fm_otcfm_CITE13176_GEO_native_full60.v1.json'
PLAN='configs/reproducibility/fm_otcfm_CITE13176_GEO_full60_after_preprocessing.2026-10-16_v1.json'
TEST='tests/test_fm_otcfm_CITE13176_GEO_native60_v1.py'
DATASET='GEO13176_QC_full_raw_RNA'
contract=base.contract


def scientific_cases():
    return [dict(id=f'otcfm_singlecell__{recipe}__{method}__{DATASET}__leave{left}__seed{seed}',
        dataset=DATASET,protocol=recipe,method=method,leaveout_timepoint=left,seed=seed,
        scientific_protocol=contract.protocol(recipe),net=dict(hidden_dims=[64,64,64],batch_norm=False,activation='selu'),
        sigma_min=1. if method=='sbcfm' else .1,ot_sampler={'cfm':None,'otcfm':'exact','sbcfm':'sinkhorn'}[method],
        expected_model_parameters=9093,native_dimensions=5,author_equivalence_certified=False,all_original_experiments_complete=False)
        for recipe in contract.PROTOCOLS for method in contract.METHODS for left in [1,2] for seed in contract.SEEDS]


def plan_registration():
    original=read(ROOT/base.original.QUEUE);base.original.verify(original)
    prep=read(ROOT/preparation.CONFIG);preparation.verify(prep)
    paths={r['path'] for r in original['source_receipts']}
    paths.update([preparation.CONFIG,base.original.QUEUE,base.MODULE.replace('.','/')+'.py',MODULE.replace('.','/')+'.py',TEST,
        'scripts/flow_matching/audit_otcfm_CITE13176_GEO_full60_replay_v1.py',
        'tests/test_fm_otcfm_CITE13176_GEO_full60_replay_v1.py',
        'scripts/flow_matching/otcfm_EB_CITE_result_integrity_v1.py',
        'scripts/flow_matching/otcfm_validation_batch_compat_v1.py'])
    paths.update(f'configs/models/fm_otcfm_CITE13176_GEO_native__{m}.v1.json' for m in contract.METHODS)
    return dict(planned_cases=scientific_cases(),required_models=60,required_preflight_combinations=12,
        preparation_configuration=preparation.CONFIG,preparation_configuration_sha256=sha(ROOT/preparation.CONFIG),
        source_queue_to_materialize=QUEUE,controller_python=sys.executable,
        state_root='experiments/runs/fm_otcfm_CITE13176_GEO_native60_deferred_v1',
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(paths)],
        source_settings=original['settings'],preserved_existing_singlecell_cases=360,
        original_scope=prep['original_scope'],historical_paper_identity_certified=False,all_original_experiments_complete=False)


def read_plan():
    p=read(ROOT/PLAN)
    if p['planned_cases']!=scientific_cases() or p['required_models']!=60 or p['required_preflight_combinations']!=12:
        raise ValueError('All60 full original-method/protocol/leaveout/seed cases required')
    for item in p['source_receipts']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Frozen deferred source/data/method contract changed')
    return p


def ready_preparation():
    p=read_plan();c=read(ROOT/p['preparation_configuration'])
    guard_path=ROOT/c['state_root']/'guard_receipt.json';report_path=Path(c['output_directory'])/preparation.previous.REPORT
    if not guard_path.exists():return None
    g=read(guard_path)
    if not g['passed'] or g['exit_code']!=0 or g['resource_guard_aborted']:
        raise ValueError('Preserved original data-preparation failure;no native fit or silent retry allowed')
    r=read(report_path)
    if (not r['passed'] or g['report_sha256']!=sha(report_path) or r['configuration_sha256']!=sha(ROOT/preparation.CONFIG)
            or r['cells']!=29394 or r['raw_RNA_features']!=58676 or r['PCA_dimensions']!=5
            or not r['original_author_NPZ_loader_full_identity_verified']
            or not r['native_author_whiten_flag_must_remain_true']
            or r['historical_Kaggle_or_paper_processed_identity_certified']
            or sha(r['target_input_for_author_loader'])!=r['target_input_sha256']):
        raise ValueError('Actual complete original raw alignment/PCA/author-loader proof required')
    return c,r,dict(preprocessing_report_path=str(report_path),preprocessing_report_sha256=sha(report_path),
        preprocessing_guard_path=str(guard_path),preprocessing_guard_sha256=sha(guard_path))


def registration():
    ready=ready_preparation()
    if ready is None:raise ValueError('Do not materialize native inputs before actual full preparation passes')
    prep,report,binding=ready;p=read_plan();parent=read(ROOT/base.original.QUEUE)
    jobs=scientific_cases()
    for j in jobs:
        minimum=min(contract.fraction_lengths(n,[.8,.1,.1])[1] for n in report['days'].values())
        batch=j['scientific_protocol']['batch_size'];effective=min(batch,minimum)
        j.update(validation_compatibility=dict(original_training_batch=batch,effective_validation_batch=effective,
            minimum_available_validation_samples=minimum,training_and_test_loaders_changed=False,
            validation_only_empty_loader_repair=effective<batch),
            data=dict(path=report['target_input_for_author_loader'],sha256=report['target_input_sha256'],cells=29394,
                days=report['days'],donor='13176',full_original_PC_dimensions=5,full_raw_RNA_features=58676,
                citation=prep['source_citation'],historical_paper_identity_certified=False),
            source_yaml=f'{base.original.AUTHOR}/runner/configs/model/{j["method"]}.yaml',
            model_config=f'configs/models/fm_otcfm_CITE13176_GEO_native__{j["method"]}.v1.json',
            artifact_directory='F:/aicoding/IIA_Data/experiments/otcfm_CITE13176_GEO_native_full60_v1/'+base.hashlib.sha256(j['id'].encode()).hexdigest()[:20])
    paths={r['path'] for r in p['source_receipts']};paths.update([PLAN,binding['preprocessing_report_path'],binding['preprocessing_guard_path']])
    return dict(jobs=jobs,required_models=60,methods=list(contract.METHODS),protocol_tracks=list(contract.PROTOCOLS),
        seeds=list(contract.SEEDS),environment_lock=base.ENVIRONMENT,python=parent['python'],numerical_threads=2,
        settings=parent['settings'],resource_lock=parent['resource_lock'],preflight_lock='experiments/runs/fm_otcfm_singlecell_readonly_replay_cpu.lock',
        state_root='experiments/runs/fm_otcfm_CITE13176_GEO_native_full60_v1',
        source_receipts=[dict(path=x,sha256=sha(ROOT/x)) for x in sorted(paths)],
        primary_metric=parent['primary_metric'],citation=parent['citation'],original_scope=p['original_scope'],
        preparation_binding=binding,preserved_existing_singlecell_cases=360,all_original_experiments_complete=False,
        boundary='All60independent publicGEO CITE13176QC/fullrawRNA cases:all29394cells,PCA5 and once-only source whitening; '
            '3native methods,2full stopping recipes,leaveouts1/2,5seeds. Complete12case real-data native preflight required. '
            'Original worker bytecode,9093parameters,couplings,sigma,optimizers,1000epoch maxima,earlystops,CPU2 and guards retained. '
            'No new PCA per model,test-label calibration,data subsampling,prior360 replacement or historical Kaggle/paper identity certification.')


def verify(q):
    if q!=registration():raise ValueError('Exact complete independent GEO CITE60 source/data/environment binding required')


def preflight_cases(q):
    selected=[j for j in q['jobs'] if j['seed']==42]
    expected={(j['protocol'],j['method'],j['leaveout_timepoint']) for j in scientific_cases()}
    if len(selected)!=12 or {(j['protocol'],j['method'],j['leaveout_timepoint']) for j in selected}!=expected:
        raise ValueError('All12 original CITE method/protocol/leaveout preflight branches required')
    return selected


def scoped_function(name):
    fn=getattr(base,name);ns=dict(vars(base),MODULE=MODULE,QUEUE=QUEUE,verify=verify,preflight_cases=preflight_cases)
    ns['worker']=FunctionType(base.worker.__code__,ns,base.worker.__name__,base.worker.__defaults__)
    if name=='worker':return ns['worker']
    if name in ['run_preflight','preflight_worker']:
        replacements={30:12};strings={'native30_preflight':'native12_preflight','all30':'all12','full150':'full60','all150':'all60','EB_CITE':'CITE13176_GEO'}
    else:replacements={150:60};strings={'all150':'all60','Full150':'Full60','full150':'full60','all30':'all12','EB_CITE':'CITE13176_GEO'}
    for helper in ['run_preflight','preflight_worker']:
        if helper!=name:ns[helper]=lambda *a,_name=helper,**kw:scoped_function(_name)(*a,**kw)
    node=next(n for n in ast.parse(Path(base.__file__).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name==name)
    class Scope(ast.NodeTransformer):
        def visit_keyword(self,n):
            self.generic_visit(n)
            if n.arg=='full150_source_queue_sha256':n.arg='full60_source_queue_sha256'
            if n.arg=='native30_report_sha256':n.arg='native12_report_sha256'
            return n
        def visit_Constant(self,n):
            if type(n.value) is int and n.value in replacements:return ast.copy_location(ast.Constant(replacements[n.value]),n)
            if isinstance(n.value,str):
                text=n.value
                for a,b in strings.items():text=text.replace(a,b)
                return ast.copy_location(ast.Constant(text),n)
            return n
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Scope().visit(node)],type_ignores=[])),str(base.__file__),'exec'),ns)
    return ns[name]


def deferred_controller():
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    api=install_facades();p=read_plan();b=ROOT/p['state_root'];b.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](b/'controller.lock')
    if owner is None:raise RuntimeError('Existing full60 deferred controller preserved')
    if (b/'status.json').exists():owner.close();raise FileExistsError('Preserve every previous deferred attempt')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),required_models=60,all_original_experiments_complete=False)
    def update(**v):state.update(v,captured_utc=datetime.now(timezone.utc).isoformat());write(b/'status.json',state)
    try:
        while ready_preparation() is None:
            update(state='waiting_actual_full_CITE_preprocessing_and_successful_guard');time.sleep(10)
        q=registration();exclusive(ROOT/QUEUE,q)
        from scripts.flow_matching import audit_otcfm_CITE13176_GEO_full60_replay_v1 as replay
        rc=replay.registration();exclusive(ROOT/replay.CONFIG,rc)
        rb=ROOT/rc['state_root'];rb.mkdir(parents=True,exist_ok=False)
        command=[p['controller_python'],'-B','-X','utf8','-u','-m',replay.MODULE]
        with (rb/'controller.stdout.log').open('xb') as out,(rb/'controller.stderr.log').open('xb') as err:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        exclusive(rb/'launch.json',dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command,
            configuration=replay.CONFIG,configuration_sha256=sha(ROOT/replay.CONFIG)))
        update(state='actual_preparation_passed_native60_and_full_replay_started',native_queue_sha256=sha(ROOT/QUEUE),replay_pid=child.pid)
        scoped_function('controller')(q)
        update(state='native60_attempts_terminal_full_replay_and_other_scope_remain')
    finally:owner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register-plan',action='store_true');parser.add_argument('--job-id');parser.add_argument('--preflight-all',action='store_true');a=parser.parse_args()
    if a.register_plan:exclusive(ROOT/PLAN,plan_registration());return
    if a.job_id or a.preflight_all:
        q=read(ROOT/QUEUE);verify(q)
        if a.preflight_all:scoped_function('preflight_worker')(q)
        else:scoped_function('worker')(q,next(j for j in q['jobs'] if j['id']==a.job_id))
    else:deferred_controller()


if __name__=='__main__':main()
