"""Keep full native training and verify restored model/optimizer/all RNG on GPU."""
import argparse
import ast
import inspect

from scripts.flow_matching import run_full_image_training_execution_recovery_v3 as original


def equal_restored(left, right, torch):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict): return left.keys()==right.keys() and all(equal_restored(left[k],right[k],torch) for k in left)
    if isinstance(left,(tuple,list)): return type(left)==type(right) and len(left)==len(right) and all(equal_restored(a,b,torch) for a,b in zip(left,right))
    return left==right


def verify_native_resume(model, optimizer, old, order, torch, np, attempt):
    import random
    if not equal_restored(model.state_dict(),old['model'],torch) or not equal_restored(optimizer.state_dict(),old['optimizer'],torch):
        raise ValueError('Every original native model and optimizer variable must restore bitwise on GPU')
    state=old['rng'];current=np.random.get_state();cuda=torch.cuda.get_rng_state_all()
    if (random.getstate()!=state['python'] or current[0]!=state['numpy'][0]
            or not np.array_equal(current[1],state['numpy'][1]) or current[2:]!=state['numpy'][2:]
            or not torch.equal(torch.get_rng_state(),state['torch_cpu'].cpu())
            or not torch.equal(order.get_state(),state['order'].cpu())
            or len(cuda)!=len(state['torch_cuda']) or len(cuda)!=1
            or any(not torch.equal(a.cpu(),b.cpu()) for a,b in zip(cuda,state['torch_cuda']))):
        raise ValueError('Complete original Python/NumPy/CPU/CUDA/order RNG restoration differs')
    if (not torch.backends.cuda.matmul.allow_tf32 or not torch.backends.cudnn.allow_tf32
            or torch.backends.cudnn.benchmark or not torch.backends.cudnn.deterministic):
        raise ValueError('All original declared native training backend flags must remain')
    original.write_json(attempt/'verified_full_native_GPU_checkpoint_restore.json',dict(
        completed_original_epochs=old['completed_epochs'],completed_original_optimizer_updates=old['global_steps'],
        original_required_epochs=2000,remaining_original_epochs=2000-old['completed_epochs'],
        full_native_parameters=sum(p.numel() for p in model.parameters()),
        all_native_model_and_optimizer_variables_bitwise_restored=True,
        Python_NumPy_CPU_all_CUDA_and_order_RNG_bitwise_restored=True,
        original_declared_BF16_TF32_cuDNN_training_policy_retained=True,
        additional_optimizer_updates_executed_by_restore_verification=0,
        no_capacity_probe_weights_loaded=True,benchmark_performance=False,all_original_experiments_complete=False))


def training_ast(path):
    tree=original.training_ast(path)
    match=ast.dump(ast.parse("restore_rng(old['rng'], torch, np, order)").body[0]);changes=0
    class Observe(ast.NodeTransformer):
        def generic_visit(self,node):
            nonlocal changes
            super().generic_visit(node)
            for attribute in ('body','orelse','finalbody'):
                body=getattr(node,attribute,None)
                if not isinstance(body,list):continue
                updated=[]
                for statement in body:
                    updated.append(statement)
                    if ast.dump(statement)==match:
                        updated+=ast.parse('_verify_full_resume(model,optimizer,old,order,torch,np,attempt)').body
                        changes+=1
                setattr(node,attribute,updated)
            return node
    tree=Observe().visit(tree)
    if changes!=1:raise ValueError('Exactly one original complete-state restoration observation required')
    return ast.fix_missing_locations(tree)


def run_ast():
    tree=ast.parse(inspect.getsource(original.run));changes=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict' and any(k.arg=='_verify_execution_hook' for k in node.keywords):
            node.keywords.append(ast.keyword(arg='_verify_full_resume',value=ast.Name(id='verify_native_resume',ctx=ast.Load())))
            changes+=1
    if changes!=1:raise ValueError('Only the isolated execution namespace gains the full GPU restore observer')
    return ast.fix_missing_locations(tree)


def run(config_path,job_path,attempt_directory,binding_path):
    config=original.read(original.ROOT/config_path)
    original.verify_receipts(config['source_receipts'])
    if job_path==config['saved401_resume']['job_configuration']:
        cpu=original.read(original.ROOT/config['saved401_resume']['CPU_proof']['path'])
        original.verify_receipts([config['saved401_resume']['CPU_proof'],config['saved401_resume']['archive']])
        if not cpu['passed'] or cpu['completed_original_epochs']!=401 or cpu['CUDA_initialized']:
            raise ValueError('Actual complete401 CPU checkpoint restoration required')
        job=original.read(original.ROOT/job_path)
        path=original.ROOT/job['output_directory']/'resume.pt'
        if original.sha(path)!=config['saved401_resume']['archive']['sha256']:
            raise ValueError('First resumed job must restore the exact verified and archived401epoch state')
    namespace=dict(original.__dict__,training_ast=training_ast,verify_native_resume=verify_native_resume)
    exec(compile(run_ast(),__file__,'exec'),namespace)
    namespace['run'](config_path,job_path,attempt_directory,binding_path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--recovery-config',required=True);parser.add_argument('--config',required=True)
    parser.add_argument('--attempt-directory',required=True);parser.add_argument('--capacity-binding',required=True)
    args=parser.parse_args();run(args.recovery_config,args.config,args.attempt_directory,args.capacity_binding)
