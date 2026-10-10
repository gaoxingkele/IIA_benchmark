"""Paired full-length real-FRED S4 convolution/recurrence diagnostic, not performance."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import random

from scripts.flow_matching import run_ls4_original_v1 as native
from scripts.flow_matching.giflow_native_protocol import write_json
from scripts.flow_matching.ls4_cauchy_fallback_repair_v1 import install


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();root=native.ROOT;settings=native.read(root/args.config)
    for item in settings['source_receipts']:
        if native.sha(root/item['path'])!=item['sha256']:raise ValueError('Frozen paired diagnostic source differs')
    queue=native.read(root/settings['original_queue']);native.verify(queue)
    job=next(j for j in queue['jobs'] if j['id']==settings['canonical_job'])
    config=native.read(root/job['model_config'])
    target=native.workspace(queue,settings['artifact_directory'])
    _,resolved=native.prepare_config(queue,config,target)
    torch,np,datasets=native.source_imports(target);torch.set_num_threads(2)
    torch.manual_seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed'])
    data,data_record=native.checked_data(config,datasets,torch,np,resolved.data)
    # All 107 original full trajectories and row split checked above. No sequence truncation.
    x=data['test_dataloader'].dataset.tensors[0][:2].transpose(1,2).repeat(1,64,1)
    s4=importlib.import_module('models.s4');raw_fallback=s4.cauchy_naive
    correction=native.read(root/settings['correction_config']);records=[]
    for repaired in (False,True):
        s4.cauchy_naive=raw_fallback
        receipt=install(s4,correction,root) if repaired else dict(algorithmic_correction=False)
        torch.manual_seed(config['seed'])
        layer=s4.S4(d_model=64,d_state=64,bidirectional=False,dropout=0.,transposed=True).eval()
        checksum=hashlib.sha256()
        for name,value in sorted(layer.state_dict().items()):
            checksum.update(name.encode());checksum.update(value.detach().cpu().numpy().tobytes())
        with torch.no_grad():
            convolution,_=layer(x);layer.setup_step(mode='dense')
            state=layer.default_state(len(x));steps=[]
            for t in range(x.shape[-1]):
                value,state=layer.step(x[:,:,t],state);steps.append(value)
            recurrent=torch.stack(steps,dim=-1)
        error=float((convolution-recurrent).abs().max())
        records.append(dict(repaired=repaired,initial_state_sha256=checksum.hexdigest(),
            shape=list(x.shape),max_abs_error=error,
            equivalent_at_1e_4=bool(torch.allclose(convolution,recurrent,rtol=1e-4,atol=1e-4)),
            backend=receipt))
    if records[0]['initial_state_sha256']!=records[1]['initial_state_sha256']:
        raise ValueError('Paired diagnostic did not use identical initial model')
    if records[0]['equivalent_at_1e_4'] or not records[1]['equivalent_at_1e_4']:
        raise ValueError('Observed fallback defect/correction not demonstrated on real full-length data')
    if torch.cuda.is_initialized():raise ValueError('CPU paired diagnostic initialized CUDA')
    write_json(root/settings['output'],dict(passed=True,benchmark_performance=False,
        diagnostic_only=True,original_queue=settings['original_queue'],
        original_queue_sha256=native.sha(root/settings['original_queue']),
        canonical_job=job['id'],data_record=data_record,cases=records,
        raw_sources_unchanged=True,captured_utc=datetime.now(timezone.utc).isoformat(),
        boundary='Two full 728-point real FRED trajectories test native S4 math consistency; all original 107 trajectories audited. This is not trained generation performance.'))
    print(json.dumps(records))


if __name__=='__main__':main()
