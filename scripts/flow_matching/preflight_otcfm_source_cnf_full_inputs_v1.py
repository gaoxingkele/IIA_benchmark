"""Full real-input likelihood/gradient checks before new CNF baseline training."""
import argparse
from datetime import datetime,timezone
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive
from scripts.flow_matching import run_otcfm_singlecell_native_v4 as source

MODULE='scripts.flow_matching.preflight_otcfm_source_cnf_full_inputs_v1'
CONFIG='configs/reproducibility/fm_otcfm_source_cnf_all12_full_input_preflight.v1.json'
MODEL='configs/models/fm_otcfm_source_cnf_leaveout_corrected.v1.json'
ADAPTER='src/iia_benchmark/models/otcfm_cnf_leaveout_v1.py'
TEST='tests/test_fm_otcfm_source_cnf_full_input_preflight_v1.py'
QUEUES=[source.QUEUE,'configs/experiments/fm_otcfm_EB_CITE_native_full150.v2.json',
    'configs/experiments/fm_otcfm_EB100_code_referenced_native_full90.v1.json']


def registration():
    datasets={}
    for path in QUEUES:
        q=read(ROOT/path)
        for j in q['jobs']:
            if j['method']=='cfm' and j['protocol']=='released_yaml' and j['seed']==42:
                if j['dataset'] in datasets and datasets[j['dataset']]!=j['data']:raise ValueError('Ambiguous original full input')
                datasets[j['dataset']]=j['data']
    cases=[dict(id=dataset+'__leftout'+str(left),dataset=dataset,data=data,leftout_timepoint=left)
        for dataset,data in sorted(datasets.items()) for left in range(1,len(data['days'])-1)]
    if len(datasets)!=5 or len(cases)!=12:raise ValueError('All five current original input tracks and12interior leaveouts required')
    base=read(ROOT/QUEUES[-1]);profile=read(ROOT/MODEL)
    files=[MODULE.replace('.','/')+'.py',MODEL,ADAPTER,TEST,'tests/test_fm_otcfm_cnf_leaveout_v1.py',
        'scripts/flow_matching/otcfm_validation_batch_compat_v1.py',*QUEUES,
        f'{source.AUTHOR}/runner/src/models/cfm_module.py',f'{source.AUTHOR}/runner/configs/model/cnf.yaml',
        f'{source.AUTHOR}/runner/configs/model/cfm.yaml']
    return dict(cases=cases,required_full_input_cases=12,distinct_original_input_tracks=5,model_configuration=MODEL,
        source_runtime_queue=source.QUEUE,environment_lock=base['environment_lock'],python=base['python'],
        numerical_threads=2,settings=base['settings'],resource_lock=base['resource_lock'],
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in files],seed=42,dimensions=5,
        train_val_test_split=[.8,.1,.1],training_batch=1024,validation_only_minimum_batch_repair=True,
        augmented_solver=dict(method='dopri5',atol=1e-5,rtol=1e-5,sensitivity='autograd'),
        optimizer=profile['optimizer'],expected_parameters=9093,optimizer_updates=0,
        state_root='experiments/runs/fm_otcfm_source_cnf_all12_full_input_preflight_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/otcfm_source_cnf_all12_full_input_preflight_v1',
        paper_citation=profile['citation'],original_scope=base['original_scope'],all_original_experiments_complete=False,
        boundary='Real-data preflight only for the source-derived plain CNF class with the separately tested reversed-index correction. '
            'All original cells/PC inputs,all12leftout branches,full1024training batch,exact likelihood trace and5dimensions retained. '
            'This explicitly assembled runtime supplies missing source NeuralODE/partial_solver constructor bindings;it is not certified original author experiment composition. '
            'Source CNF optimizer definition is AdamW0.001/weight_decay0.01,not silently copied from CFM. '
            'Only validation batch shrinks to the real minimum split,never training/test populations. '
            'One full training-batch objective/gradient and validation loss per case,zero optimizer updates. '
            'All full CNF training/replay/metrics,seeds/data tracks and true standalone TrajectoryNet,Reg.CNF,DSB baselines remain required. '
            'No preflight is benchmark performance,no original source/result is overwritten,and original native32GiB guards remain.')


def verify(c):
    if c!=registration():raise ValueError('All12complete real-data cases and exact source/profile/resource bindings required')


def worker(c):
    from functools import partial
    torch,pl,distribution,instantiate,OmegaConf=source.source_runtime(read(ROOT/c['source_runtime_queue']))
    from torchdyn.core import NeuralODE
    from src.models.cfm_module import CNFLitModule
    from scripts.flow_matching.otcfm_validation_batch_compat_v1 import install
    spec=importlib.util.spec_from_file_location('_iia_source_cnf_leaveout_adapter',ROOT/ADAPTER)
    adapter=importlib.util.module_from_spec(spec);sys.modules[spec.name]=adapter;spec.loader.exec_module(adapter)
    records=[]
    for case in c['cases']:
        data=case['data'];path=ROOT/data['path']
        if sha(path)!=data['sha256']:raise ValueError('Untouched complete original input changed')
        pl.seed_everything(c['seed'],workers=True)
        dm=distribution.CustomTrajectoryDataModule(system=str(path),train_val_test_split=tuple(c['train_val_test_split']),
            max_dim=c['dimensions'],whiten=True,batch_size=c['training_batch'],num_workers=0,pin_memory=False,seed=c['seed'])
        counts={str(level):len(x) for level,x in zip(dm.ulabels,dm.timepoint_data)}
        if counts!=data['days'] or sum(counts.values())!=data['cells'] or dm.dim!=5:raise ValueError('Every original cell and all timepoints required')
        minimum_val=min(len(parts[1]) for parts in dm.split_timepoint_data)
        repair=install(dm,min(c['training_batch'],minimum_val))
        config=OmegaConf.merge(OmegaConf.load(ROOT/source.AUTHOR/'runner/configs/model/cfm.yaml'),
            OmegaConf.load(ROOT/source.AUTHOR/'runner/configs/model/cnf.yaml'))
        config.partial_solver.ode_solver=c['augmented_solver']['method']
        config.partial_solver.atol=c['augmented_solver']['atol'];config.partial_solver.rtol=c['augmented_solver']['rtol']
        model=adapter.build_corrected_cnf(CNFLitModule,datamodule=dm,
            net=instantiate(config.net),optimizer=instantiate(config.optimizer),augmentations=instantiate(config.augmentations),
            partial_solver=instantiate(config.partial_solver),leaveout_timepoint=case['leftout_timepoint'],plot=False,
            neural_ode=partial(NeuralODE,solver=c['augmented_solver']['method'],atol=c['augmented_solver']['atol'],
                rtol=c['augmented_solver']['rtol'],sensitivity=c['augmented_solver']['sensitivity']))
        if sum(p.numel() for p in model.parameters())!=9093:raise ValueError('Full original three64-unit CNF velocity network required')
        class Observe(torch.nn.Module):
            def __init__(self,inner):super().__init__();self.inner=inner;self.starts=[]
            def forward(self,x,ts):self.starts.append(x.detach().clone());return self.inner(x,ts)
        model.aug_node=Observe(model.aug_node)
        batch=[x.clone().detach().requires_grad_(True) for x in next(iter(dm.train_dataloader()))]
        if any(len(x)!=1024 for x in batch):raise ValueError('Never lower the full native1024training batch')
        reg,loss=model.step(batch,training=True);objective=reg+loss;objective.backward()
        if not torch.isfinite(objective) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Real original likelihood and all model gradients must be finite')
        expected=[i for i in reversed(range(len(batch))) if i!=case['leftout_timepoint']]
        if len(model.aug_node.starts)!=len(expected) or not all(torch.equal(value,batch[i].detach()) for value,i in zip(model.aug_node.starts,expected)):
            raise ValueError('Actual original NLL integrator must omit exactly the requested observation index')
        for index,x in enumerate(batch):
            if index==case['leftout_timepoint']:
                if x.grad is not None and torch.count_nonzero(x.grad):raise ValueError('Requested heldout observations leaked into the likelihood')
            elif x.grad is None or not torch.isfinite(x.grad).all() or not torch.count_nonzero(x.grad):
                raise ValueError('Every retained real observation timepoint must enter the likelihood gradient')
        model.aug_node.starts.clear();validation=next(iter(dm.val_dataloader()))
        with torch.no_grad():vr,vl=model.step(validation,training=False)
        if not torch.isfinite(vr+vl):raise ValueError('Original augmented CNF validation likelihood must be finite')
        optimizer=model.configure_optimizers()
        if type(optimizer) is not torch.optim.AdamW or any(g['lr']!=.001 or g['weight_decay']!=.01 for g in optimizer.param_groups):
            raise ValueError('Keep the original CNF model optimizer definition')
        records.append(dict(id=case['id'],full_data_cells=data['cells'],full_timepoint_counts=counts,dimensions=5,parameters=9093,
            training_batch=1024,validation_batch=repair['effective_validation_batch'],source_validation_batch_repair=repair,
            actual_training_timepoint_indices=expected,requested_leftout=case['leftout_timepoint'],
            objective=float(objective.detach()),validation_objective=float((vr+vl).detach()),
            requested_leftout_gradient_zero=True,all_retained_timepoint_gradients_nonzero_finite=True,
            optimizer_updates=0,benchmark_performance_claim=False))
        write(ROOT/c['state_root']/'progress.json',dict(completed_cases=len(records),required_cases=12,current_id=case['id']))
        del optimizer,model,dm,batch,validation,reg,loss,objective,vr,vl;gc.collect()
    out=Path(c['artifact_root']);out.mkdir(parents=True,exist_ok=False)
    exclusive(out/'full12_native_likelihood_preflight.json',dict(passed=True,records=records,required_cases=12,
        configuration_sha256=sha(ROOT/CONFIG),optimizer_updates=0,full_training_complete=False,
        original_Source_files_modified=False,historical_paper_equivalence_certified=False,all_original_experiments_complete=False))


def controller(c):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=install_facades();base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Full CNF preflight already has an owner')
    if (base/'status.json').exists():owner.close();raise FileExistsError('Preserve every original preflight attempt')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),required_full_input_cases=12,active_pid=None,all_original_experiments_complete=False)
    def update(value):state.update(value,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],update):
            verify(c)
            command=[c['python'],'-B','-X','utf8','-u','-m',MODULE,'--worker']
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
                update(dict(state='running_all12_full_input_CNF_preflight',active_pid=child.pid,exact_child_handle=handle))
                code,usage=capped_wait(child,c,api,update)
            report=Path(c['artifact_root'])/'full12_native_likelihood_preflight.json'
            passed=code==0 and not usage['resource_guard_aborted'] and report.exists() and read(report)['passed']
            exclusive(base/'guard_receipt.json',dict(passed=passed,exit_code=code,**usage,exact_child_handle=handle,report_sha256=sha(report) if report.exists() else None))
            update(dict(state='full12_preflight_passed_training_still_pending' if passed else 'preflight_failed_preserved',active_pid=None))
    finally:owner.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.register:exclusive(ROOT/CONFIG,registration());return
    c=read(ROOT/CONFIG);verify(c)
    if a.worker:worker(c)
    else:controller(c)


if __name__=='__main__':main()
