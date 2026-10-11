"""Diagnose every full SMAP input using all five preserved float32 checkpoints.

These are precision-changed re-inference diagnostics, never float64-trained
benchmark results or completion of the new from-scratch training controls.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.flow_matching.run_light_controller import ROOT, digest
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait


def verify(config):
    for entry in config['source_receipts']:
        if digest(ROOT / entry['path']) != entry['sha256']:
            raise ValueError('Frozen complete precision diagnostic source differs')
    if len(config['cases']) != 5 or {c['job']['seed'] for c in config['cases']} != {1103,1104,1105,1106,1107}:
        raise ValueError('Every preserved SMAP checkpoint and original seed required')


def worker(config):
    import numpy as np
    import torch
    from scripts.flow_matching.run_tsad_experiment import covering_windows, align_scores
    from iia_benchmark.models.mtsad_detectors import TranADDetector
    from iia_benchmark.models.fm_tranad_reference_float64_v1 import ReferenceFloat64TranADDetector
    torch.set_num_threads(config['original_cpu_threads'])
    target=ROOT/config['output_directory']
    if target.exists():raise FileExistsError('Preserve every prior full diagnostic attempt')
    target.mkdir(parents=True)
    manifest=json.loads((ROOT/config['dataset_manifest']).read_text(encoding='utf-8'))
    dataset=next(d for d in manifest['datasets'] if d['id']=='smap')
    if digest(ROOT/dataset['input_npz'])!=dataset['input_sha256']:raise ValueError('Frozen full SMAP input differs')
    records=[]
    with np.load(ROOT/dataset['input_npz'],allow_pickle=False) as inputs:
        for case in config['cases']:
            job=case['job'];checkpoint=torch.load(ROOT/case['checkpoint'],map_location='cpu',weights_only=False)
            model_config=json.loads((ROOT/job['model_config']).read_text(encoding='utf-8'))
            if (checkpoint['configuration']!=model_config or checkpoint['overrides']!=job['parameter_overrides']
                    or checkpoint['window']!=job['window']):raise ValueError('Original checkpoint scientific identity differs')
            weights=checkpoint['modules']['_model']
            if not all(torch.isfinite(v).all() for v in weights.values()):raise ValueError('Original trained checkpoint already nonfinite')
            parameters=dict(model_config['parameters']['parameters'],epochs=model_config['parameters']['epochs'],seed=job['seed'],device='cpu')
            old=TranADDetector(**parameters);new=ReferenceFloat64TranADDetector(**parameters)
            features=dataset['entities'][0]['features']
            for model,double in [(old,False),(new,True)]:
                model._device=torch.device('cpu');model._model=model._build(features)
                if double:model._model=model._model.double()
                model._model.load_state_dict(weights,strict=True)
            payload={};coverage=[]
            for entity in dataset['entities']:
                prefix=entity['array_prefix'];parts={}
                for split in ['train','validation','test']:
                    values=inputs[prefix+'_'+split];windows,starts=covering_windows(values,job['window'])
                    before=old.score(windows);after=new.score(windows)
                    if after.shape!=(len(windows),) or not np.isfinite(after).all():raise ValueError('Float64 did not cover every real window finitely')
                    scores=align_scores(after,starts,len(values),job['window'])
                    payload[prefix+'_'+split]=scores
                    parts[split]=dict(full_points=len(values),full_windows=len(windows),input_max_abs=float(np.max(np.abs(values))),
                        float32_nonfinite_windows=int((~np.isfinite(before)).sum()),float64_nonfinite_windows=int((~np.isfinite(after)).sum()))
                labels=inputs[prefix+'_labels'];payload[prefix+'_labels']=labels
                if len(labels)!=entity['test_points']:raise ValueError('Full original label coverage differs')
                coverage.append(dict(entity=entity['entity_id'],splits=parts,test_points=len(labels)))
            arrays=target/(job['id']+'__diagnostic_scores_from_original_float32_weights.npz')
            np.savez_compressed(arrays,**payload)
            record=dict(original_float32_job_id=job['id'],seed=job['seed'],checkpoint=case['checkpoint'],checkpoint_sha256=case['checkpoint_sha256'],
                original_weights_all_finite=True,full_registered_entities=len(coverage),full_registered_test_points=sum(c['test_points'] for c in coverage),
                float32_nonfinite_windows=sum(s['float32_nonfinite_windows'] for c in coverage for s in c['splits'].values()),
                float64_all_real_train_validation_test_windows_finite=True,complete_full_registered_timestamp_coverage=True,
                diagnostic_scores=arrays.relative_to(ROOT).as_posix(),diagnostic_scores_sha256=digest(arrays),entity_coverage=coverage,
                new_float64_training_done=False,benchmark_result=False,author_equivalence_certified=False)
            records.append(record)
            (target/'progress.json').write_text(json.dumps(dict(completed_original_checkpoint_diagnostics=len(records),required=5,latest=record),indent=2)+'\n',encoding='utf-8',newline='\n')
            print(json.dumps(dict(completed=len(records),required=5,seed=job['seed'],float32_nonfinite_windows=record['float32_nonfinite_windows'],full_registered_entities=len(coverage))),flush=True)
    report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),all5preserved_original_checkpoints_full_registered_inputs_diagnosed=True,
        records=records,original_cpu_threads=config['original_cpu_threads'],new_from_scratch_float64_training_done=False,
        additional_training_seed_slots=0,benchmark_result=False,author_equivalence_certified=False,strict_TAB_result=False,
        all_original_experiments_complete=False,boundary='Full precision-changed re-inference on all original trained float32 SMAP weights and every original train/validation/test point. Demonstrates or rejects numerical overflow repair, not author float64 training equivalence. New25full from-scratch controls remain required; no original model/data/results altered.')
    (target/'precision_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();config=json.loads((ROOT/args.config).read_text(encoding='utf-8'));verify(config)
    if args.worker:return worker(config)
    api=resource_api();base=ROOT/config['controller_directory'];base.mkdir(parents=True,exist_ok=True)
    status=dict(pid=os.getpid(),config_sha256=digest(ROOT/args.config),state='starting',active_pid=None)
    def update(values):
        status.update(values)
        temporary=base/'status.tmp';temporary.write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8',newline='\n');temporary.replace(base/'status.json')
    with api['resource_slot'](ROOT/config['readonly_resource_lock'],config['settings'],update):
        command=[config['python'],'-X','utf8','-u','-m','scripts.flow_matching.audit_tranad_float64_existing_checkpoints_v1','--config',args.config,'--worker']
        with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
            child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),CUDA_VISIBLE_DEVICES='-1',PYTHONUTF8='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            update(dict(state='replaying_full_checkpoints',active_pid=child.pid))
            code,usage=capped_wait(child,config,api,update)
        update(dict(state='completed' if code==0 else 'failed_preserved',active_pid=None,exit_code=code,resources=usage,benchmark_result=False))
        if code:raise RuntimeError('Full diagnostic failed; preserve all existing artifacts')


if __name__=='__main__':main()
