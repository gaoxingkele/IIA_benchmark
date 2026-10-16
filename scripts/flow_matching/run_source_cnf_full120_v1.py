"""Full source-derived CNF training on all five current original data tracks.

Both source model-default and experiment-override decay settings stay separate.
Neither assembled trajectory recipe is certified as a historical paper run.
"""
import argparse
import ast
import copy
from datetime import datetime,timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import FunctionType,SimpleNamespace

from scripts.flow_matching import preflight_otcfm_source_cnf_full_inputs_v3 as preflight
from scripts.flow_matching import run_otcfm_singlecell_native_v4 as original
from scripts.flow_matching import run_otcfm_EB_CITE_native150_v2 as template
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive

MODULE='scripts.flow_matching.run_source_cnf_full120_v1'
QUEUE='configs/experiments/fm_source_cnf_full120.v1.json'
MODEL='configs/models/fm_source_cnf_full_training.v1.json'
TEST='tests/test_fm_source_cnf_full120_v1.py'
ENVIRONMENT=original.ENVIRONMENT
identity=original.identity
PROTOCOLS=['source_model_default_decay','source_experiment_decay']


def scientific_cases():
    upstream=read(ROOT/preflight.CONFIG)
    cases=[]
    for case in upstream['cases']:
        for protocol in PROTOCOLS:
            for seed in [42,43,44,45,46]:
                recipe=original.contract.protocol('released_yaml')
                recipe.update(batch_size=1024,weight_decay=.01 if protocol==PROTOCOLS[0] else 1e-5,
                    ode_solver='dopri5',ode_atol=1e-5,ode_rtol=1e-5,
                    boundary='Source-derived trajectory CNF;full source CNF batch1024/max1000/validation10,default patience100checks;explicit solver assembly,not historical experiment composition')
                cid=f'source_cnf__{protocol}__{case["dataset"]}__leave{case["leftout_timepoint"]}__seed{seed}'
                minimum=min(original.contract.fraction_lengths(n,[.8,.1,.1])[1] for n in case['data']['days'].values())
                cases.append(dict(id=cid,dataset=case['dataset'],protocol=protocol,method='source_derived_cnf',
                    leaveout_timepoint=case['leftout_timepoint'],seed=seed,data=case['data'],
                    scientific_protocol=recipe,expected_model_parameters=9093,native_dimensions=5,
                    model_config=MODEL,artifact_directory='F:/aicoding/IIA_Data/experiments/source_cnf_full120_v1/'+hashlib.sha256(cid.encode()).hexdigest()[:20],
                    validation_compatibility=dict(original_training_batch=1024,effective_validation_batch=min(1024,minimum),
                        minimum_available_validation_samples=minimum,training_and_test_loaders_changed=False,
                        validation_only_empty_loader_repair=minimum<1024),
                    original_source_native_execution=False,source_derived_repaired_execution=True,
                    author_equivalence_certified=False,all_original_experiments_complete=False))
    return cases


def registration():
    upstream=read(ROOT/preflight.CONFIG);preflight.verify(upstream)
    paths=[MODULE.replace('.','/')+'.py',MODEL,TEST,preflight.CONFIG,
        'src/iia_benchmark/models/otcfm_cnf_manual_training_v1.py',
        'scripts/flow_matching/audit_source_cnf_full120_v1.py',
        'scripts/flow_matching/otcfm_EB_CITE_result_integrity_v1.py',
        'scripts/flow_matching/audit_otcfm_EB_CITE_full150_replay_v1.py',
        template.MODULE.replace('.','/')+'.py',
        f'{original.AUTHOR}/runner/configs/experiment/cnf.yaml',
        f'{original.AUTHOR}/runner/configs/callbacks/default.yaml']
    paths += [p['path'] for p in upstream['source_receipts']]
    jobs=scientific_cases()
    if len(jobs)!=120 or len({j['id'] for j in jobs})!=120:raise ValueError('All12leaveouts x2source parameter tracks x5training seeds required')
    return dict(jobs=jobs,required_models=120,required_complete_five_seed_cohorts=24,
        original_input_tracks=5,interior_leaveout_branches=12,protocol_tracks=PROTOCOLS,seeds=[42,43,44,45,46],
        prerequisite_configuration=preflight.CONFIG,environment_lock=upstream['environment_lock'],
        python=upstream['python'],numerical_threads=2,settings=upstream['settings'],resource_lock=upstream['resource_lock'],
        state_root='experiments/runs/fm_source_cnf_full120_v1',primary_metric='test/t_out/1-Wasserstein',
        lifecycle_artifact_root='F:/aicoding/IIA_Data/experiments/source_cnf_full120_v1_framework_lifecycle',
        evaluation_inference_mode=False,
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(set(paths))],
        citation=upstream['paper_citation'],original_scope=upstream['original_scope'],
        boundary='All five complete available original cell/PC input tracks and all12interior leaveouts,source model decay.01 and experiment override1e-5,each seeds42..46. '
            'Original CNF exact-trace likelihood,three64-unit SELU network,AdamW.001,full1024training batch,max1000epochs,validation every10epochs and default patience100checks. '
            'Missing solver/NeuralODE assembly is explicitly dopri5/autograd/atol=rtol1e-5. Both forward-generation and full test populations remain original. '
            'Reversed leaveout-index and scalar trace-shape corrections plus all-endpoint gradient accumulation are disclosed isolated source repairs. '
            'Each full batch uses every retained endpoint and exactly one optimizer step. No endpoint minibatch reduction,AMP,gradient clipping or budget shortening. '
            'Validation-only empty-loader repair retains every original train/test observation. Whitening/source min-size loader and validation including leftout remain as release code. '
            'Training may start only after all12actual full-input preflights and their successful process guard,and native framework lifecycle fixtures. '
            'Evaluation uses no_grad rather than inference_mode because exact trace requires nested input autograd;all derivative and likelihood equations stay unchanged. '
            'Full fresh-checkpoint replay independently solves complete W1 costs and requires every native scalar within2e-6. '
            'Source-derived plain CNF is not standalone TrajectoryNet or Reg.CNF and never certifies historical paper identity. All other baseline/method/data tasks remain required.',
        all_original_experiments_complete=False)


def verify(q):
    if q!=registration():raise ValueError('Exact full120 source CNF/data/seed/parameter/resource contract required')


def prerequisite(q):
    c=read(ROOT/q['prerequisite_configuration']);preflight.verify(c)
    base=ROOT/c['state_root'];gp=base/'guard_receipt.json'
    if not gp.exists():return None
    guard=read(gp);report_path=Path(c['artifact_root'])/'full12_native_likelihood_preflight.json'
    if not guard['passed'] or guard['exit_code']!=0 or guard['resource_guard_aborted']:
        raise ValueError('Full-input preflight failed; preserve it and do not train')
    report=read(report_path)
    if (not report['passed'] or report['configuration_sha256']!=sha(ROOT/preflight.CONFIG)
            or guard['report_sha256']!=sha(report_path) or len(report['records'])!=12
            or {r['id'] for r in report['records']}!={r['id'] for r in c['cases']}):
        raise ValueError('All12actual full-input conditions and unchanged successful report hash required')
    for row in report['records']:
        if not (row['training_batch']==1024 and row['parameters']==9093 and row['optimizer_updates']==0
                and row['requested_leftout_gradient_zero'] and row['all_retained_timepoint_gradients_nonzero_finite']):
            raise ValueError('Every actual full native likelihood/gradient condition required')
    return dict(guard_path=str(gp),guard_sha256=sha(gp),report_path=str(report_path),report_sha256=sha(report_path))


def adapter(path,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value


def trainer_facade(pl):
    def Trainer(**kwargs):
        if 'inference_mode' in kwargs:raise ValueError('CNF exact likelihood requires the frozen no_grad evaluation context')
        return pl.Trainer(inference_mode=False,**kwargs)
    return SimpleNamespace(Trainer=Trainer)


def instantiate_case(q,job):
    from functools import partial
    torch,pl,distribution,instantiate,OmegaConf=original.source_runtime(q)
    from torchdyn.core import NeuralODE
    from src.models.cfm_module import CNFLitModule
    from scripts.flow_matching.otcfm_validation_batch_compat_v1 import install
    pl.seed_everything(job['seed'],workers=True)
    recipe=job['scientific_protocol']
    if sha(ROOT/job['data']['path'])!=job['data']['sha256']:raise ValueError('Untouched complete input changed')
    dm=distribution.CustomTrajectoryDataModule(system=str(ROOT/job['data']['path']),train_val_test_split=(.8,.1,.1),
        max_dim=5,whiten=True,batch_size=1024,num_workers=0,pin_memory=False,seed=job['seed'])
    counts={str(level):len(x) for level,x in zip(dm.ulabels,dm.timepoint_data)}
    if counts!=job['data']['days'] or sum(counts.values())!=job['data']['cells'] or dm.dim!=5:
        raise ValueError('Every original cell and complete original selected PC input required')
    if install(dm,job['validation_compatibility']['effective_validation_batch'])!=job['validation_compatibility']:
        raise ValueError('Exact validation-only loader repair required')
    config=OmegaConf.merge(OmegaConf.load(ROOT/original.AUTHOR/'runner/configs/model/cfm.yaml'),
        OmegaConf.load(ROOT/original.AUTHOR/'runner/configs/model/cnf.yaml'))
    config.optimizer.weight_decay=recipe['weight_decay']
    config.partial_solver.ode_solver=recipe['ode_solver'];config.partial_solver.atol=recipe['ode_atol'];config.partial_solver.rtol=recipe['ode_rtol']
    leaveout=adapter('src/iia_benchmark/models/otcfm_cnf_leaveout_v1.py','_iia_full_cnf_leaveout')
    model=leaveout.build_corrected_cnf(CNFLitModule,datamodule=dm,net=instantiate(config.net),
        optimizer=instantiate(config.optimizer),augmentations=instantiate(config.augmentations),
        partial_solver=instantiate(config.partial_solver),leaveout_timepoint=job['leaveout_timepoint'],plot=False,
        neural_ode=partial(NeuralODE,solver='dopri5',atol=1e-5,rtol=1e-5,sensitivity='autograd'))
    adapter('src/iia_benchmark/models/otcfm_cnf_trace_shape_v1.py','_iia_full_cnf_trace').install_model(model)
    adapter('src/iia_benchmark/models/otcfm_cnf_endpoint_backward_v1.py','_iia_full_cnf_endpoints').install_model(model,CNFLitModule,leaveout.step_tree)
    adapter('src/iia_benchmark/models/otcfm_cnf_manual_training_v1.py','_iia_full_cnf_training').install(model)
    if sum(p.numel() for p in model.parameters())!=9093:raise ValueError('Complete original9093parameter network required')
    optimizer=model.configure_optimizers()
    if type(optimizer) is not torch.optim.AdamW or any(g['lr']!=.001 or g['weight_decay']!=recipe['weight_decay'] for g in optimizer.param_groups):
        raise ValueError('Actual source-parameter-track optimizer required')
    return torch,trainer_facade(pl),dm,model


def lifecycle_worker(q):
    """Synthetic framework integration only;never a shortened benchmark run."""
    torch,pl,*_=original.source_runtime(q)
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader,TensorDataset
    manual=adapter('src/iia_benchmark/models/otcfm_cnf_manual_training_v1.py','_iia_CNF_lifecycle_manual')
    root=Path(q['lifecycle_artifact_root']);root.mkdir(parents=True,exist_ok=False)
    records=[]
    for number,(points,left,decay) in enumerate([(4,1,.01),(5,3,1e-5)]):
        pl.seed_everything(42,workers=True)
        class Fixture(pl.LightningModule):
            def __init__(self):
                super().__init__();self.parameter=torch.nn.Parameter(torch.linspace(.1,.5,5))
                self.save_hyperparameters(dict(leaveout_timepoint=left));self.is_trajectory=True
                self._iia_endpoint_backward_installed=True;manual.install(self)
            def step(self,batch,training,endpoint_index):
                value=(batch[endpoint_index]*self.parameter).sin().square().mean()+self.parameter.square().mean()
                return self.parameter.sum()*0,value
            def configure_optimizers(self):return torch.optim.AdamW(self.parameters(),lr=.001,weight_decay=decay)
            def validation_step(self,batch,batch_idx):
                # The original exact trace likewise needs grad during evaluation.
                with torch.set_grad_enabled(True):
                    x=batch[0].detach().requires_grad_(True)
                    trace=torch.autograd.grad((x*self.parameter).sin().sum(),x,create_graph=True)[0].sum(1).mean()
                self.log('val/loss',trace.detach(),batch_size=1024)
        model=Fixture();folder=root/str(number);folder.mkdir()
        batch=[torch.randn(2048,5) for _ in range(points)]
        loader=DataLoader(TensorDataset(*batch),batch_size=1024,shuffle=False,drop_last=True)
        checkpoint=ModelCheckpoint(dirpath=folder,monitor='val/loss',save_last=True,save_top_k=-1)
        trainer=trainer_facade(pl).Trainer(default_root_dir=folder,max_epochs=2,accelerator='cpu',devices=1,
            precision=32,logger=False,enable_progress_bar=False,callbacks=[checkpoint],num_sanity_val_steps=0)
        trainer.fit(model,train_dataloaders=loader,val_dataloaders=loader)
        saved=torch.load(folder/'last.ckpt',map_location='cpu')
        if trainer.global_step!=4 or int(saved['global_step'])!=4 or int(saved['epoch'])!=1:
            raise ValueError('Lightning manual optimizer must count exactly one update per complete batch')
        opt=saved['optimizer_states']
        if (len(opt)!=1 or any(int(v['step'])!=4 for v in opt[0]['state'].values())
                or sum(v['exp_avg'].numel() for v in opt[0]['state'].values())!=5):
            raise ValueError('Native synthetic lifecycle checkpoint must retain all actual moments and counters')
        records.append(dict(timepoints=points,leftout=left,decay=decay,full_fixture_batch=1024,
            epochs=2,actual_optimizer_updates=4,all_fixture_checkpoint_moments_verified=True,
            no_grad_evaluation_allows_exact_trace_autograd=True,benchmark_performance_claim=False))
    exclusive(root/'synthetic_native_lifecycle_proof.json',dict(passed=True,records=records,
        exact_environment_lock_sha256=sha(ROOT/q['environment_lock']),full_CNF_benchmark_conditions_completed=0,
        actual_full12_real_input_preflight_binding=prerequisite(q),all_original_experiments_complete=False))


def lifecycle_gate(q,api,base,update):
    import psutil
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    attempt=base/'native_framework_lifecycle';path=Path(q['lifecycle_artifact_root'])/'synthetic_native_lifecycle_proof.json'
    if attempt.exists() or Path(q['lifecycle_artifact_root']).exists():raise FileExistsError('Preserve previous native lifecycle attempt without automatic retry')
    with api['resource_slot'](ROOT/q['resource_lock'],q['settings'],update):
        attempt.mkdir()
        command=[q['python'],'-B','-X','utf8','-u','-m',MODULE,'--native-lifecycle']
        with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
            child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
            update(dict(state='testing_native_framework_lifecycle_not_benchmark',active_pid=child.pid,exact_child_handle=handle))
            code,usage=capped_wait(child,q,api,update)
        passed=code==0 and not usage['resource_guard_aborted'] and path.exists() and read(path)['passed']
        exclusive(attempt/'guard_receipt.json',dict(passed=passed,exit_code=code,**usage,exact_child_handle=handle,proof_sha256=sha(path) if path.exists() else None))
        if not passed:raise ValueError('Native framework integration failed; source training must not start')
        update(dict(state='native_framework_lifecycle_passed_full_training_pending',active_pid=None))


def worker(q,job):
    if prerequisite(q) is None:raise ValueError('All12full native preflights must actually pass before fitting')
    binding=lifecycle_binding(q)
    def result_write(path,value):
        value=dict(value,original_source_native_execution=False,source_derived_repaired_execution=True,
            full_input_preflight_binding=prerequisite(q),source_CNF_parameter_track=job['protocol'],
            native_framework_lifecycle_binding=binding,endpoint_gradient_accumulation=True,one_optimizer_update_per_full_training_batch=True)
        exclusive(path,value)
    f=FunctionType(template.worker.__code__,dict(template.worker.__globals__,instantiate_case=instantiate_case,
        ENVIRONMENT=ENVIRONMENT,QUEUE=QUEUE,exclusive=result_write),argdefs=template.worker.__defaults__)
    f(q,job)


def lifecycle_binding(q):
    guard_path=ROOT/q['state_root']/'native_framework_lifecycle/guard_receipt.json'
    proof_path=Path(q['lifecycle_artifact_root'])/'synthetic_native_lifecycle_proof.json'
    guard,proof=read(guard_path),read(proof_path)
    if (not guard['passed'] or guard['exit_code']!=0 or guard['resource_guard_aborted']
            or guard['proof_sha256']!=sha(proof_path) or not proof['passed']
            or proof['exact_environment_lock_sha256']!=sha(ROOT/q['environment_lock'])
            or proof['actual_full12_real_input_preflight_binding']!=prerequisite(q)
            or len(proof['records'])!=2 or any(r['actual_optimizer_updates']!=4 for r in proof['records'])):
        raise ValueError('Actual successful native framework lifecycle guard and full checkpoint proof required')
    return dict(guard_path=str(guard_path),guard_sha256=sha(guard_path),proof_path=str(proof_path),proof_sha256=sha(proof_path))


def controller(q):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=install_facades();base=ROOT/q['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None or (base/'status.json').exists():
        if owner:owner.close()
        raise FileExistsError('Never duplicate or overwrite this full source CNF attempt')
    done=[];failed={};state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None)
    def update(value):
        state.update(value,completed_ids=done,failed_preserved=failed,required_models=120,
            captured_utc=datetime.now(timezone.utc).isoformat(),all_original_experiments_complete=False)
        write(base/'status.json',state)
    try:
        while prerequisite(q) is None:
            update(dict(state='waiting_actual_full12_native_preflight',active_pid=None));time.sleep(10)
        update(dict(state='full12_native_preflight_passed',preflight_binding=prerequisite(q)))
        lifecycle_gate(q,api,base,update)
        for job in q['jobs']:
            key=Path(job['artifact_directory']).name;attempt=base/'attempts'/key
            try:
                if attempt.exists() or Path(job['artifact_directory']).exists():raise FileExistsError('Preserve every full or partial CNF attempt')
                with api['resource_slot'](ROOT/q['resource_lock'],q['settings'],update):
                    attempt.mkdir(parents=True,exist_ok=False)
                    command=[q['python'],'-B','-X','utf8','-u','-m',MODULE,'--job-id',job['id']]
                    with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
                        child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                            env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',
                                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg',WANDB_MODE='disabled'),
                            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                        handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
                        update(dict(state='running_full_source_CNF_training_test',active_pid=child.pid,exact_child_handle=handle,active_job=job['id']))
                        code,usage=capped_wait(child,q,api,update)
                    path=Path(job['artifact_directory'])/'result.json';passed=code==0 and not usage['resource_guard_aborted'] and path.exists()
                    if passed:passed=read(path)['job']==job and read(path)['status']=='completed'
                    exclusive(attempt/'guard_receipt.json',dict(passed=passed,exit_code=code,**usage,exact_child_handle=handle,result_sha256=sha(path) if path.exists() else None))
                    if not passed:raise ValueError('Full source CNF numerical attempt failed; preserve without retry')
                    done.append(job['id'])
            except Exception as error:failed[job['id']]=repr(error)
            update(dict(state='between_full_source_CNF_cases',active_pid=None))
        update(dict(state='all120_source_CNF_attempts_terminal_other_experiments_remain',active_pid=None))
    finally:owner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true');parser.add_argument('--job-id');parser.add_argument('--probe-only',action='store_true');parser.add_argument('--native-lifecycle',action='store_true');args=parser.parse_args()
    if args.register:exclusive(ROOT/QUEUE,registration());return
    q=read(ROOT/QUEUE);verify(q)
    if args.probe_only:print(json.dumps(dict(required_models=len(q['jobs']),five_seed_cohorts=24,numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)));return
    if args.native_lifecycle:lifecycle_worker(q)
    elif args.job_id:worker(q,next(j for j in q['jobs'] if j['id']==args.job_id))
    else:controller(q)


if __name__=='__main__':main()
