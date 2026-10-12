"""Observe every original SHCL update/test window without changing native math."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from types import ModuleType

from scripts.flow_matching import shcl_released_native_v1 as native


def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def solver_ast(source):
    tree=ast.parse(source); counts={name:0 for name in ('before_forward','after_step','after_epoch','after_score',
        'before_adjustment','finish_train','finish_test')}
    solver=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Solver')
    def observation(text,name):
        counts[name]+=1
        return ast.parse(text).body[0]
    def instrument(body,phase):
        result=[]
        for node in body:
            active=phase
            if isinstance(node,ast.For) and isinstance(node.iter,ast.Call) and node.iter.args:
                loader=node.iter.args[0]
                if isinstance(loader,ast.Attribute) and loader.attr in ('thre_loader','test_loader'):
                    active='threshold' if loader.attr=='thre_loader' else 'test'
            if hasattr(node,'body') and isinstance(node.body,list): node.body=instrument(node.body,active)
            if hasattr(node,'orelse') and isinstance(node.orelse,list): node.orelse=instrument(node.orelse,active)
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='data_dict' for t in node.targets):
                if ast.dump(node.value)!=ast.dump(ast.parse('self.model(input)',mode='eval').body):
                    raise ValueError('Every original forward call must remain exact')
                label='_' if active=='validation' else 'labels'
                result.append(observation('_observer.before_forward(input,'+label+','+repr(active)+')','before_forward'))
            result.append(node)
            if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.dump(node.value)==ast.dump(ast.parse(
                    'self.optimizer.step()',mode='eval').body):
                result.append(observation('_observer.after_step(self)','after_step'))
            if (isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
                    and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='torch' and node.value.func.attr=='save'):
                result.append(observation('_observer.after_epoch(self,epoch,train_loss,vali_loss)','after_epoch'))
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cri' for t in node.targets):
                result.append(observation('_observer.after_score(loss,cri,labels)','after_score'))
            if (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='gt' for t in node.targets)
                    and ast.dump(node.value)==ast.dump(ast.parse('test_labels.astype(int)',mode='eval').body)):
                result.append(observation('_observer.before_adjustment(pred,gt,thresh)','before_adjustment'))
            if isinstance(node,ast.Return) and phase=='test':
                result[-1:]=[observation('_observer.finish_test(self,results)','finish_test'),node]
        return result
    for function in solver.body:
        if isinstance(function,ast.FunctionDef) and function.name in ('train','vali','test'):
            phase={'train':'train','vali':'validation','test':'test'}[function.name]
            function.body=instrument(function.body,phase)
            if phase=='train': function.body.append(observation('_observer.finish_train(self)','finish_train'))
    if counts!=dict(before_forward=4,after_step=1,after_epoch=1,after_score=2,before_adjustment=1,finish_train=1,finish_test=1):
        raise ValueError('Exact full native training/validation/threshold/test observation points required: '+str(counts))
    return ast.fix_missing_locations(tree)


class Observer:
    def __init__(self,job,directory):
        self.job=job; self.directory=Path(directory); self.updates=0; self.epochs=[]
        self.counts={name:0 for name in ('train','validation','threshold','test')}
        self.frames=[]; self.pending=None; self.raw_prediction=None

    def rng(self):
        import torch
        import numpy as np
        return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state().clone(),
            torch_cuda=[value.clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [])

    def backend_flags(self):
        import torch
        return dict(cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_enabled=torch.backends.cudnn.enabled,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled())

    def unchanged_rng(self,before):
        import numpy as np
        import torch
        after=self.rng()
        if (before['python']!=after['python'] or before['numpy'][0]!=after['numpy'][0]
                or not np.array_equal(before['numpy'][1],after['numpy'][1]) or before['numpy'][2:]!=after['numpy'][2:]
                or not torch.equal(before['torch_cpu'],after['torch_cpu'])
                or len(before['torch_cuda'])!=len(after['torch_cuda'])
                or any(not torch.equal(a,b) for a,b in zip(before['torch_cuda'],after['torch_cuda']))):
            raise ValueError('Observation must never advance original native RNG streams')

    def progress(self):
        write(self.directory/'progress.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),job_id=self.job['id'],
            original_optimizer_updates=self.updates,forward_batches=self.counts,completed_epochs=len(self.epochs),
            required_epochs=self.job['num_epochs'],full_original_experiment_complete=False))

    def before_forward(self,inputs,labels,phase):
        self.counts[phase]+=1; self.phase=phase
        if phase in ('threshold','test'):
            before=self.rng()
            self.pending=dict(input=inputs.detach().cpu().clone(),labels=labels.detach().cpu().clone(),
                pre_forward_rng=before,backend_flags=self.backend_flags(),phase=phase,original_batch_index=self.counts[phase]-1)
            self.unchanged_rng(before)

    def after_step(self,solver):
        self.updates+=1
        if self.updates>self.job['expected_optimizer_updates']:
            raise ValueError('Original optimizer update budget exceeded')
        if self.updates%100==0: self.progress()

    def after_epoch(self,solver,epoch,train_loss,validation_loss):
        import torch
        before=self.rng(); target=self.directory/'epochs'/('epoch_'+str(epoch+1).zfill(4)); target.mkdir(parents=True,exist_ok=False)
        path=target/'full_state.pt'
        with path.open('xb') as stream:
            torch.save(dict(model=solver.model.state_dict(),optimizer=solver.optimizer.state_dict(),
                original_rng=before,completed_epoch=epoch+1,optimizer_updates=self.updates,
                backend_flags=self.backend_flags(),source_job=self.job),stream)
        row=dict(epoch=epoch+1,optimizer_updates=self.updates,train_loss=float(train_loss),
            validation_loss=float(validation_loss),full_state=str(path),full_state_sha256=native.sha(path),
            original_model_checkpoint_sha256=native.sha(Path(solver.model_save_path)/(solver.dataset+'_checkpoint.pth')))
        self.epochs.append(row); write(target/'receipt.json',row); self.unchanged_rng(before); self.progress()

    def after_score(self,loss,score,labels):
        import numpy as np
        import torch
        if self.pending is None or self.pending['phase']!=self.phase: raise ValueError('Every original test batch context required')
        before=self.rng(); target=self.directory/'frames'; target.mkdir(exist_ok=True)
        prefix=self.phase+'_'+str(self.pending['original_batch_index']).zfill(6)
        context=target/(prefix+'_context.pt'); arrays=target/(prefix+'_scores.npz')
        self.pending['post_forward_rng']=before
        with context.open('xb') as stream: torch.save(self.pending,stream)
        raw=loss.detach().cpu().numpy()
        if raw.shape!=score.shape or not np.isfinite(raw).all() or not np.isfinite(score).all():
            raise ValueError('Complete finite native source losses and softmax scores required')
        with arrays.open('xb') as stream:
            np.savez_compressed(stream,loss=raw,score=score,labels=labels.detach().cpu().numpy())
        self.frames.append(dict(phase=self.phase,batch_index=self.pending['original_batch_index'],
            shape=list(score.shape),context=str(context),context_sha256=native.sha(context),
            arrays=str(arrays),arrays_sha256=native.sha(arrays)))
        self.pending=None; self.unchanged_rng(before)

    def before_adjustment(self,prediction,truth,threshold):
        import numpy as np
        self.raw_prediction=(prediction.copy(),truth.copy(),float(threshold))
        with (self.directory/'native_unadjusted_predictions.npz').open('xb') as stream:
            np.savez_compressed(stream,prediction=prediction,truth=truth,threshold=threshold)

    def finish_train(self,solver):
        if self.updates!=self.job['expected_optimizer_updates'] or len(self.epochs)!=self.job['num_epochs']:
            raise ValueError('Every original full epoch/update and saved optimizer/RNG state required')
        if self.counts['train']!=self.updates or self.counts['validation']!=self.job['expected_validation_batches']:
            raise ValueError('All source train and full validation batches required')
        self.progress()
        write(self.directory/'result.json',dict(passed=True,job=self.job,optimizer_updates=self.updates,
            epochs=self.epochs,forward_batches=self.counts,original_training_complete=not self.job.get('synthetic_fixture',False),
            synthetic_operator_only_not_benchmark_performance=self.job.get('synthetic_fixture',False),
            full_original_tests_complete=False,checkpoint_replay_complete=False,paper_equivalence_certified=False))

    def finish_test(self,solver,results):
        import numpy as np
        from sklearn.metrics import precision_recall_fscore_support,accuracy_score
        full={}
        for phase in ('threshold','test'):
            entries=[row for row in self.frames if row['phase']==phase]
            expected=self.job['expected_'+phase+'_windows']
            if sum(row['shape'][0] for row in entries)!=expected: raise ValueError('All native windows and complete batch archives required')
            arrays=[]; labels=[]
            for row in entries:
                with np.load(row['arrays'],allow_pickle=False) as archive:
                    arrays.append(archive['score']); labels.append(archive['labels'])
            full[phase]=np.concatenate(arrays,axis=0).reshape(-1)
            if phase=='test': full['labels']=np.concatenate(labels,axis=0).reshape(-1).astype(int)
        threshold=float(np.percentile(full['threshold'],100-self.job['anormly_ratio']))
        pred=(full['test']>threshold).astype(int); truth=full['labels']
        saved_pred,saved_truth,saved_threshold=self.raw_prediction
        if not np.array_equal(pred,saved_pred) or not np.array_equal(truth,saved_truth) or threshold!=saved_threshold:
            raise ValueError('Independent complete score/threshold replay must equal original pre-adjustment predictions')
        precision,recall,f1,_=precision_recall_fscore_support(truth,pred,average='binary')
        for key,value in [('accuracy',accuracy_score(truth,pred)),('precision',precision),('recall',recall),('f_score',f1),('threshold',threshold)]:
            if float(results[key])!=float(value): raise ValueError('Original native scalar differs from complete saved arrays: '+key)
        write(self.directory/'result.json',dict(passed=True,job=self.job,original_results=results,
            full_source_score_frames=self.frames,forward_batches=self.counts,
            full_native_test_prefix_points=len(truth),original_native_labels_and_pre_PA_predictions_verified=True,
            original_percentile_threshold_recomputed_exactly=True,full_native_test_complete=not self.job.get('synthetic_fixture',False),
            synthetic_operator_only_not_benchmark_performance=self.job.get('synthetic_fixture',False),
            native_nonoverlap_dropped_tail_points=self.job['source_dropped_tail_rows'],
            strict_full_point_TAB=False,checkpoint_replay_complete=False,paper_equivalence_certified=False))


def install(model,observer):
    source=native.verify(model); sys.path.insert(0,str(source))
    if 'solver' in sys.modules: raise ValueError('Use a fresh isolated native worker; never patch a live solver')
    module=ModuleType('solver'); module.__file__=str(source/'solver.py'); module.__dict__['_observer']=observer
    sys.modules['solver']=module
    exec(compile(solver_ast((source/'solver.py').read_text(encoding='utf-8')),str(source/'solver.py'),'exec'),module.__dict__)
    return module
