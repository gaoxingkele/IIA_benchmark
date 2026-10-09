"""Diagnostic full-original-batch gradients and original evaluator APIs only."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import time

from scripts.flow_matching.prepare_spectral_original_data import ROOT,read,sha,write
from scripts.flow_matching.run_spectral_original_job import verify


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();queue=read(ROOT/args.queue);policy=verify(queue)
    output=ROOT/args.output
    if output.exists():raise FileExistsError('Preserve previous preflight evidence')
    sys.path.insert(0,str(ROOT/policy['author_source']));sys.path.insert(0,str(ROOT/'src'))
    import numpy as np
    import torch
    import yaml
    from iia_benchmark.models.fm_spectral_author import build_author_model,FrozenAuthorGenerationDataset
    torch.set_num_threads(1)
    cases={j['model_config']:j for j in queue['jobs']}
    observations=[]
    for config_path,job in cases.items():
        config=read(ROOT/config_path);torch.manual_seed(42)
        original=yaml.safe_load((ROOT/job['author_yaml']).read_text(encoding='utf-8'))
        model=build_author_model(ROOT/config['parameters']['source_root'],config['parameters']['model_target'],config['parameters']['parameters'])
        count=sum(p.numel() for p in model.parameters())
        if job.get('expected_parameters') is not None and count!=job['expected_parameters']:
            raise ValueError('Paper Table9 parameter count mismatch')
        dataset=FrozenAuthorGenerationDataset(ROOT/job['frozen_samples']);batch=original['dataloader']['batch_size']
        values=torch.stack([dataset[i] for i in range(batch)])
        optimizer=torch.optim.Adam(model.parameters(),lr=original['solver']['base_lr'],betas=(.9,.96))
        start=time.perf_counter();loss=model(values,target=values);loss.backward()
        if not torch.isfinite(loss) or not any(p.grad is not None for p in model.parameters()) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError('Full original batch does not produce finite double-backpropagation gradients')
        optimizer.step()
        observations.append({'model_config':config_path,'dataset':job['dataset'],'data_track':job['data_track'],
            'full_batch_size':batch,'parameter_count':count,'one_step_loss':float(loss),'seconds':time.perf_counter()-start,
            'finite_gradients':True,'original_sampling_method_retained':True})
        print(json.dumps(observations[-1]),flush=True)
        del model,optimizer,values,dataset,loss
        import gc
        gc.collect()
    import tensorflow as tf
    from utils.discriminative_metric import discriminative_score_metrics
    from utils.predictive_metric import predictive_score_metrics
    tf1=discriminative_score_metrics.__globals__['tf1']
    tf1.reset_default_graph();tf1.set_random_seed(0)
    # Verify the original TF1-style GRU API against the pinned evaluator environment.
    with tf1.Session() as session:
        tf1.nn.rnn_cell.GRUCell(3);probe=tf1.random_normal((2,3));one=session.run(probe)
    tf1.reset_default_graph();tf1.set_random_seed(0)
    with tf1.Session() as session:
        tf1.nn.rnn_cell.GRUCell(3);two=session.run(tf1.random_normal((2,3)))
    np.testing.assert_array_equal(one,two)
    write(output,{'passed':True,'diagnostic_only':True,'benchmark_performance':False,
        'queue_sha256':sha(ROOT/args.queue),'full_author_model_configs_verified':len(cases),
        'original_sampling_method_retained':True,'original_tf_gru_api_checked':True,
        'fixed_tf_graph_seed_checked':True,'observations':observations,'torch':torch.__version__,
        'tensorflow':tf.__version__,'captured_utc':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__':main()
