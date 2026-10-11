"""Run frozen full2000epoch FM/WT jobs with verified execution-only recovery.

Keep every original loss, permutation, native optimizer, batch, epoch and
checkpoint boundary. Extend Windows IO paths and enable the separately tested
native nonreentrant activation recomputation. Capacity-probe weights are never
loaded. Original failed attempts and downstream canonical paths are preserved.
"""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT,read,sha,verify_receipts
from scripts.flow_matching.full_image_training_execution_recovery_v2 import (
    enable_native_activation_recomputation,native_path,save_checkpoint,write_json)


def training_ast(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main']
    if len(tree.body)!=1:raise ValueError('Exactly one frozen full training entry point required')
    main=tree.body[0]
    matches=[i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and len(n.targets)==1
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='model'
        and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='to'
        and isinstance(n.value.func.value,ast.Call) and isinstance(n.value.func.value.func,ast.Name)
        and n.value.func.value.func.id=='FlowMismatchUNet']
    if len(matches)!=1:raise ValueError('Only the verified complete original U-Net construction may receive the execution hook')
    main.body.insert(matches[0]+1,ast.parse('_verify_execution_hook(model,model_config,environment,job,attempt_directory=args.attempt_directory)').body[0])
    return ast.fix_missing_locations(tree)


def verify_capacity(binding):
    for item in binding['proof_receipts']:verify_receipts([item])
    if {record['objective'] for record in binding['cases']}!={'FM','WT'}:
        raise ValueError('Both actual native full-batch objective capacity proofs are required')
    for record in binding['cases']:
        proof=read(record['proof_path']);terminal=read(record['terminal_path'])
        if (proof['case']['objective']!=record['objective'] or proof['actual_batch_shape']!=[32,3,128,128]
                or proof['full_model_parameters']!=45289539 or proof['native_recomputed_blocks']!={'ResBlock':54,'AttentionBlock':29}
                or not proof['full_batch_forward_backward_optimizer_and_checkpoint_IO_passed']
                or proof['actual_optimizer_updates']!=1 or not proof['capacity_probe_only']
                or proof['full_original_training_executed'] or proof['benchmark_performance']
                or not proof['bfloat16'] or not proof['TF32'] or terminal['returncode']!=0
                or terminal['resource_usage']['resource_guard_aborted']):
            raise ValueError('Successful full native batch32 CUDA/BF16/TF32 execution is required before full training')
        verify_receipts([proof['checkpoint']])
    return True


def run(config_path,job_path,attempt_directory,binding_path):
    config=read(ROOT/config_path);verify_receipts(config['source_receipts'])
    binding=read(ROOT/binding_path)
    if binding['recovery_configuration_sha256']!=sha(ROOT/config_path):
        raise ValueError('Capacity dispatch must bind the full recovery configuration')
    verify_capacity(binding)
    registered={entry['path']:objective for objective in ('FM','WT')
        for entry in read(ROOT/config['original_runtimes'][objective]['queue'])['jobs']}
    if job_path not in registered:raise ValueError('Only an unchanged original full-cohort job may run')
    objective=registered[job_path]
    if objective=='FM':
        from scripts.flow_matching import run_flow_mismatching_full_training_v1 as original
    else:
        from scripts.flow_matching import run_wt_full_image_training_v1 as original
    job=read(ROOT/job_path)
    attempt=Path(attempt_directory).resolve()
    output=(ROOT/job['output_directory']).resolve()
    if not attempt.is_relative_to(output/'attempts') or not attempt.is_dir():
        raise ValueError('A new attempt within the original canonical output is required')
    for failure in config['original_failure_receipts']:verify_receipts([failure])

    def hook(model,model_config,environment,actual_job,attempt_directory):
        import torch
        if actual_job!=job:raise ValueError('The frozen original job must remain unchanged')
        expected=next(x for x in environment['full_architectures']
            if x['dataset']==job['dataset'] and x['variant']==job['variant'])
        count=sum(p.numel() for p in model.parameters())
        if count!=expected['parameter_count'] or model_config['parameters']!=expected['architecture_parameters']:
            raise ValueError('Every parameter of the registered full dataset/conditioning architecture is required')
        blocks=enable_native_activation_recomputation(model)
        if blocks!={'ResBlock':54,'AttentionBlock':29}:raise ValueError('All native full-backbone blocks must be retained')
        write_json(attempt/'verified_full_training_execution_adapter.json',dict(
            objective=objective,original_job=job_path,original_job_sha256=sha(ROOT/job_path),
            full_parameter_count=count,native_recomputed_blocks=blocks,
            complete_original_epochs=2000,batch_size=32,bfloat16=True,TF32=True,
            original_dataset_images_per_epoch=job['expected_training_images_per_epoch'],
            original_total_optimizer_updates=job['expected_optimizer_steps'],
            native_checkpoint_mode='nonreentrant_preserve_rng_state',
            original_full_train_epoch_and_loss_functions_unchanged=True,
            native_capacity_proof_binding=binding,capacity_probe_weights_loaded=False,
            all_raw_and_source_files_read_only=True,benchmark_performance=False))

    import sys
    proxy=SimpleNamespace(**vars(original.os))
    proxy.link=lambda source,destination:original.os.link(native_path(source),native_path(destination))
    namespace=dict(vars(original),write=write_json,save_checkpoint=save_checkpoint,os=proxy,_verify_execution_hook=hook)
    source=Path(original.__file__)
    exec(compile(training_ast(source),str(source),'exec'),namespace)
    sys.argv=[str(source),'--config',job_path,'--attempt-directory',str(attempt)]
    namespace['main']()
    verify_receipts(config['source_receipts'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery-config',required=True);parser.add_argument('--config',required=True)
    parser.add_argument('--attempt-directory',required=True);parser.add_argument('--capacity-binding',required=True)
    args=parser.parse_args()
    run(args.recovery_config,args.config,args.attempt_directory,args.capacity_binding)
