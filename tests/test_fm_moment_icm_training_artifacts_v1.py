import json
import os
from pathlib import Path
import subprocess
import pytest
from scripts.flow_matching import moment_icm_training_artifacts_v1 as p

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('batches,items,sources',[(2,3,1),(39479,80850189,1764),(39478,80850189,2086),(39479,80850188,2086)])
def test_scientific_checkpoint_identity_cannot_certify_a_reduced_full_epoch(tmp_path,batches,items,sources):
    job=dict(id='full_epoch',epochs=1,expected_batches=batches,expected_items=items,sources=sources,scientific=True)
    with pytest.raises(ValueError,match='smaller corpus'):p.TrainingArtifacts(tmp_path/'state',job,'a'*64)
    assert not (tmp_path/'state').exists()


NATIVE=r'''
from pathlib import Path
from types import SimpleNamespace
import copy,json,sys
import torch
from scripts.flow_matching import moment_icm_masked_training_v1 as train
from scripts.flow_matching import moment_icm_training_artifacts_v1 as store
from scripts.flow_matching import prepare_moment_icm_full_epoch_plan_v1 as full
from scripts.flow_matching import moment_icm_disk_epoch_plan_v1 as disk
from scripts.flow_matching import moment_icm_pile_cache_reader_v1 as data
torch.set_num_threads(2)
root=Path.cwd();temporary=Path(sys.argv[1]);source=root/'experiments/runs/mtsad_protocol_audit/sources/moment_research/original'
names=['moment/utils/masking.py','moment/utils/optims.py','moment/utils/utils.py','moment/tasks/base.py','moment/tasks/pretrain.py']
pins={n:train.digest(source/n) for n in names};Masking,_=train.load_source_symbols(source,pins)
cls=train.core_class(root/'src/iia_benchmark/models');core=cls(str(source),task='reconstruction',d_model=16,num_layers=2,heads=2,d_ff=32,dropout=.1)
model=train.masked_model(core,Masking,gradient_checkpointing=True)
sources,_=full.complete_sources();s=copy.deepcopy(next(s for s in sources if s['family']=='classification' and s['native_channels']==2))
s['training_base_cases_or_windows']=3;s['items']=3
disk.build(temporary/'plan',[s],2,13);reader=data.PileCacheReader([s],epoch_seed=13)
loader=train.CompleteEpochLoader(disk.DiskEpochPlan(temporary/'plan'),reader)
job=dict(id='real_cache_three_case_fixture',epochs=1,expected_batches=2,expected_items=3,sources=1,scientific=False)
artifacts=store.TrainingArtifacts(temporary/'artifacts',job,'a'*64)
args=SimpleNamespace(max_epoch=1,max_opt_steps=5000000,checkpoint_path=str(artifacts.checkpoint_root),use_amp=False,
 optimizer_name='AdamW',init_lr=1e-4,weight_decay=.05,lr_decay_rate=.9,warmup_lr=1e-5,warmup_steps=1000,min_lr=1e-5,
 set_input_mask=True,torch_dtype='bfloat16',max_norm=5.,debug=False,log_interval=100000,checkpoint_interval=1)
task=train.SourceTrainingTask(model,loader,args,source,pins,artifacts,device='cpu',save_callback=artifacts.save_periodic,external_callback=artifacts.defer_external)
task.train();reader.close()
assert artifacts.logged_steps==task.audited_batches==2 and len(artifacts.checkpoints)==len(artifacts.pending_evaluations)==2
assert all(r['status']=='NOT_EXECUTED_PENDING' and not r['actual_downstream_evaluation_complete'] for r in artifacts.pending_evaluations)
assert all(c['phase']=='source_periodic_pre_scheduler' for c in artifacts.checkpoints)
# Final proof cannot be written after a truncated or unaudited update count.
task.audited_batches=1
try:artifacts.finalize(task,loader)
except ValueError:pass
else:raise AssertionError('partial training certified')
assert not (artifacts.root/'complete_training_state.json').exists()
task.audited_batches=2
before=store.capture_rng();receipt=artifacts.finalize(task,loader);after=store.capture_rng()
assert store.exact(before,after) and receipt['passed'] and receipt['scope']['items']==3
assert len(receipt['checkpoints'])==3 and not receipt['all_external_evaluations_complete']
assert not receipt['independent_model_reinference_complete'] and not receipt['benchmark_or_paper_performance_certified']
path=Path(receipt['final_checkpoint']['path']);payload=torch.load(path,map_location='cpu',weights_only=False)
assert store.sha(path)==receipt['final_checkpoint']['sha256'] and payload['job']==job and payload['optimizer_updates']==2
assert payload['phase']=='final_post_scheduler'
assert payload['optimizer_current_learning_rates']==[task.optimizer.param_groups[0]['lr']]
assert payload['scheduler_state']=={k:v for k,v in vars(task.lr_scheduler).items() if k!='optimizer'}
assert store.exact(payload['model_state_dict'],model.state_dict()) and store.exact(payload['rng'],before)
changed=copy.deepcopy(payload);changed['rng']['torch_cpu'][0]^=1
assert not store.exact(payload,changed)
state=next(iter(task.optimizer.state.values()));old=state['step'].clone();state['step']-=1
try:store.audit_optimizer(task.optimizer,2)
except ValueError:pass
else:raise AssertionError('reduced optimizer state accepted')
state['step'].copy_(old)
try:artifacts.save_periodic(model,artifacts.checkpoint_root/'wrong',2,task.optimizer,task.scaler,task.lr_scheduler)
except ValueError:pass
else:raise AssertionError('unbound output path accepted')
assert all(train.digest(source/name)==value for name,value in pins.items())
record=dict(passed=True,actual_source_training_fixture_cases=3,actual_source_fixture_optimizer_updates=2,
 complete_saved_model_optimizer_AMP_scheduler_RNG_exact=True,checkpoint_IO_consumes_no_training_RNG=True,
 complete_source_partial_final_batch_kept=True,deferred_external_evaluation_still_incomplete=True,
 native_model_reinference_or_full_scientific_training_executed=False,fixture_is_benchmark_performance=False)
(temporary/'native_artifact_check.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record))
'''


def test_native_complete_source_callbacks_final_state_roundtrip_and_false_completion_rejections(tmp_path):
    config=json.loads((ROOT/'configs/experiments/fm_moment_pretrained_execution.v1.json').read_text(encoding='utf-8'))
    result=subprocess.run([str(ROOT/config['python']),'-B','-X','utf8','-c',NATIVE,str(tmp_path)],cwd=ROOT,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+';'+str(ROOT/'src'),CUDA_VISIBLE_DEVICES='-1',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout[-2000:]+result.stderr[-5000:]
    assert json.loads(result.stdout.strip().splitlines()[-1])['passed']
