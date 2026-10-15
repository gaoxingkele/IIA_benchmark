import ast
import json
import os
from pathlib import Path
import subprocess
import pytest
from scripts.flow_matching import moment_icm_masked_training_v1 as p

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'experiments/runs/mtsad_protocol_audit/sources/moment_research/original'
TRAIN=SOURCE/'moment/tasks/pretrain.py'


def test_training_ast_preserves_complete_source_operations_except_three_disclosed_insertions():
    node=p.controlled_train_node(TRAIN,p.digest(TRAIN))
    assert ast.unparse(next(n for n in ast.walk(node) if isinstance(n,ast.While)).test)==p.epoch.PAPER_CONDITION
    body=next(n for n in ast.walk(node) if isinstance(n,ast.For)).body
    text='\n'.join(ast.unparse(n) for n in body)
    assert text.count('self.target_validity(')==text.count('self.model.finish_backward()')==text.count('self.audit_gradients_and_loss(')==1
    assert text.index('self.scaler.scale(loss).backward()')<text.index('self.model.finish_backward()')<text.index('self.scaler.unscale_(')
    assert text.index('nn.utils.clip_grad_norm_(')<text.index('self.audit_gradients_and_loss(')<text.index('self.scaler.step(')


def test_training_adapter_refuses_another_upstream_version():
    with pytest.raises(ValueError,match='SHA256'):p.controlled_train_node(TRAIN,'0'*64)


NATIVE=r'''
from pathlib import Path
import copy,json,sys
from types import SimpleNamespace
import torch
import numpy as np
from scripts.flow_matching import moment_icm_masked_training_v1 as p
from scripts.flow_matching import prepare_moment_icm_full_epoch_plan_v1 as full
from scripts.flow_matching import moment_icm_disk_epoch_plan_v1 as disk
from scripts.flow_matching import moment_icm_pile_cache_reader_v1 as data
MomentICMT5=p.core_class(Path.cwd()/'src/iia_benchmark/models')
torch.set_num_threads(2)
root=Path.cwd();source=root/'experiments/runs/mtsad_protocol_audit/sources/moment_research/original'
names=['moment/utils/masking.py','moment/utils/optims.py','moment/utils/utils.py','moment/tasks/base.py','moment/tasks/pretrain.py']
pins={n:p.digest(source/n) for n in names};Masking,namespace=p.load_source_symbols(source,pins)
bad=dict(pins);bad['moment/utils/masking.py']='0'*64
try:p.load_source_symbols(source,bad)
except ValueError:pass
else:raise AssertionError('changed source accepted')
torch.manual_seed(73)
# Full paper-shaped 4x256/4head/1024FF reconstruction model, not a trained
# benchmark: one synthetic CPU forward/backward solely checks checkpoint math.
core=MomentICMT5(str(source),task='reconstruction',dropout=0.)
plain=p.masked_model(copy.deepcopy(core),Masking,gradient_checkpointing=False)
fixed=p.masked_model(copy.deepcopy(core),Masking,gradient_checkpointing=True)
broken=copy.deepcopy(core);broken.encoder.gradient_checkpointing_enable()
values=torch.randn(1,8,256);padding=torch.ones(1,256,dtype=torch.long)
mask=torch.ones_like(padding);mask[:,128:192]=0
plain.train();fixed.train();broken.train()
a=plain(values,padding,mask);b=fixed(values,padding,mask)
old=broken(values,input_mask=padding,reconstruction_mask=mask)
assert torch.equal(a.reconstruction,b.reconstruction) and torch.equal(old,a.reconstruction)
assert all(layer.channels==8 for layer in fixed.core.icm_layers)
try:fixed(values,padding,mask)
except RuntimeError:pass
else:raise AssertionError('overlapping checkpoint context accepted')
((a.reconstruction-values)**2).mean().backward();plain.finish_backward()
((b.reconstruction-values)**2).mean().backward();fixed.finish_backward()
((old-values)**2).mean().backward()
base=dict(plain.core.named_parameters());repaired=dict(fixed.core.named_parameters());buggy=dict(broken.named_parameters())
diff=[];bugdiff=[]
for name,param in base.items():
 if param.grad is None:
  assert repaired[name].grad is None and buggy[name].grad is None
 else:
  assert torch.isfinite(param.grad).all() and torch.isfinite(repaired[name].grad).all()
  diff.append(float((param.grad-repaired[name].grad).abs().max()))
  bugdiff.append(float((param.grad-buggy[name].grad).abs().max()))
assert max(diff)<=1e-6,(max(diff),max(bugdiff))
assert max(bugdiff)>1e-4,(max(diff),max(bugdiff))
assert not fixed.pending_backward and all(layer.channels==1 for layer in fixed.core.icm_layers)
assert len(fixed.core.icm_layers)==4 and sum(layer.beta.numel() for layer in fixed.core.icm_layers)==16
# Source RNG and mask exactly match; constructing/wrapping adds no random draw.
torch.manual_seed(91);expected=Masking(mask_ratio=.3,patch_len=8).generate_mask(values,padding)
torch.manual_seed(91);fixed.eval()
with torch.no_grad():actual=fixed(values,padding)
assert torch.equal(expected,actual.pretrain_mask) and not fixed.pending_backward
try:fixed(values,padding,torch.zeros_like(padding))
except ValueError as e:assert 'No retained source observations' in str(e)
else:raise AssertionError('implicit all-empty normalization fallback')
# Keep the source dropout0.1 as well: checkpoint recomputation must preserve
# both the real channel grouping and Torch RNG, not just a dropout-free case.
drop_core=MomentICMT5(str(source),task='reconstruction',d_model=16,num_layers=2,heads=2,d_ff=32,dropout=.1)
drop_plain=p.masked_model(copy.deepcopy(drop_core),Masking,gradient_checkpointing=False)
drop_fixed=p.masked_model(copy.deepcopy(drop_core),Masking,gradient_checkpointing=True)
torch.manual_seed(111);out1=drop_plain(values,padding,mask);((out1.reconstruction-values)**2).mean().backward();drop_plain.finish_backward()
torch.manual_seed(111);out2=drop_fixed(values,padding,mask);((out2.reconstruction-values)**2).mean().backward();drop_fixed.finish_backward()
assert torch.equal(out1.reconstruction,out2.reconstruction)
for a_param,b_param in zip(drop_plain.parameters(),drop_fixed.parameters()):
 if a_param.grad is None:assert b_param.grad is None
 else:assert torch.equal(a_param.grad,b_param.grad)
# Validity is per channel/time, intersects source random-hidden mask. All-true
# control must preserve original scalar loss and every prediction gradient.
task=object.__new__(p.SourceTrainingTask);task.active_targets=0;task.zero_target_batches=0
valid=torch.ones(1,8,256,dtype=torch.bool);valid[:,0,128:192]=False
original_mask=(padding*(1-mask)).unsqueeze(1).repeat(1,8,1)
filtered=task.target_validity(original_mask,SimpleNamespace(valid_targets=valid))
assert int(filtered.sum())==7*64 and not filtered[:,0].any()
prediction=torch.randn_like(values,requires_grad=True);truth=torch.randn_like(values)
original_loss=(original_mask*(prediction-truth)**2).nansum()/(original_mask.nansum()+1e-7)
alltrue=task.target_validity(original_mask,SimpleNamespace(valid_targets=torch.ones_like(valid)))
control_loss=(alltrue*(prediction-truth)**2).nansum()/(alltrue.nansum()+1e-7)
assert torch.equal(original_loss,control_loss)
g1=torch.autograd.grad(original_loss,prediction,retain_graph=True)[0]
g2=torch.autograd.grad(control_loss,prediction)[0];assert torch.equal(g1,g2)
try:task.target_validity(original_mask,SimpleNamespace(valid_targets=valid.float()))
except ValueError:pass
else:raise AssertionError('nonbool targets accepted')
# Execute the complete isolated upstream train body on 3 actual cached cases,
# batch2 then the final partial1. This is a narrow plumbing fixture only.
sources,_=full.complete_sources();s=copy.deepcopy(next(s for s in sources if s['family']=='classification' and s['native_channels']==2))
s['training_base_cases_or_windows']=3;s['items']=3
temporary=Path(sys.argv[1]);disk.build(temporary/'plan',[s],2,13)
reader=data.PileCacheReader([s],epoch_seed=13);loader=p.CompleteEpochLoader(disk.DiskEpochPlan(temporary/'plan'),reader)
small=MomentICMT5(str(source),task='reconstruction',d_model=16,num_layers=2,heads=2,d_ff=32,dropout=0.)
model=p.masked_model(small,Masking,gradient_checkpointing=True)
logs=[];saved=[];external=[]
logger=SimpleNamespace(name='real_cache_three_case_fixture',log=lambda value:logs.append(dict(value)),alert=lambda **kw:None)
args=SimpleNamespace(max_epoch=1,max_opt_steps=5000000,checkpoint_path=str(temporary/'checkpoints'),use_amp=False,
 optimizer_name='AdamW',init_lr=1e-4,weight_decay=.05,lr_decay_rate=.9,warmup_lr=1e-5,warmup_steps=1000,min_lr=1e-5,
 set_input_mask=True,torch_dtype='bfloat16',max_norm=5.,debug=False,log_interval=100000,checkpoint_interval=1)
def save(model,path,step,optimizer,scaler,scheduler):
 assert all(int(v['step'])==step for v in optimizer.state.values() if 'step' in v)
 file=Path(path)/f'fixture_{step}.pt';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict()),file)
 loaded=torch.load(file,map_location='cpu');assert all(torch.equal(loaded['model'][name],value.cpu()) for name,value in model.state_dict().items())
 saved.append(step)
task=p.SourceTrainingTask(model,loader,args,source,pins,logger,device='cpu',save_callback=save,external_callback=lambda path,step,device:external.append(step))
task.train();scope=loader.complete();reader.close()
assert scope['batches']==2 and scope['items']==3 and task.audited_batches==2 and saved==external==[1,2]
assert len(logs)==2 and not model.pending_backward
assert task.active_targets>0 and all(np.isfinite(x['step_train_loss']) for x in logs)
# Original one-epoch scheduler warms up then remains at init_lr within epoch0;
# do not silently replace it with cosine decay over batch slots.
assert abs(task.optimizer.param_groups[0]['lr']-(1e-5+(1e-4-1e-5)*2/1000))<1e-15
try:list(loader)
except RuntimeError:pass
else:raise AssertionError('implicit repeated epoch accepted')
# A complete zero-valid-target batch retains its optimizer/scheduler slot;
# invalid targets are not trained, and no sample or update is silently dropped.
class OneBatch:
 drop_last=False
 def __len__(self):return 1
 def __iter__(self):
  yield SimpleNamespace(timeseries=values,input_mask=padding,valid_targets=torch.zeros_like(valid))
zero_model=p.masked_model(copy.deepcopy(small),Masking,gradient_checkpointing=True)
zero_args=copy.deepcopy(args);zero_args.checkpoint_path=str(temporary/'zero_target_checkpoints')
zero_logs=[];zero_logger=SimpleNamespace(name='zero_target_fixture',log=lambda x:zero_logs.append(x),alert=lambda **kw:None)
zero=p.SourceTrainingTask(zero_model,OneBatch(),zero_args,source,pins,zero_logger,device='cpu',save_callback=save,external_callback=lambda *args:None)
zero.train()
assert zero.active_targets==0 and zero.zero_target_batches==zero.audited_batches==1
assert zero_logs[0]['step_train_loss']==0. and not zero_model.pending_backward
assert all(int(v['step'])==1 for v in zero.optimizer.state.values() if 'step' in v)
assert all(p.digest(source/name)==value for name,value in pins.items())
print(json.dumps(dict(passed=True,full_paper_architecture_gradient_checkpointing_checked=True,
 repaired_gradient_max_abs_difference=max(diff),original_reset_bug_gradient_max_abs_difference=max(bugdiff),
 source_dropout_checkpoint_rng_and_gradients_bitwise_preserved=True,
 complete_source_body_real_cache_fixture_batches=2,fixture_items=3,fixture_is_benchmark_performance=False,
 zero_valid_target_batch_keeps_source_optimizer_slot=True,actual_full_pretraining_epochs_executed=0)))
'''


def test_native_full_architecture_gradients_masks_and_complete_source_train_body(tmp_path):
    environment=json.loads((ROOT/'configs/experiments/fm_moment_pretrained_execution.v1.json').read_text(encoding='utf-8'))
    result=subprocess.run([str(ROOT/environment['python']),'-B','-X','utf8','-c',NATIVE,str(tmp_path)],
        cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+';'+str(ROOT/'src'),CUDA_VISIBLE_DEVICES='-1',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout[-2000:]+result.stderr[-6000:]
    record=json.loads(result.stdout.strip().splitlines()[-1])
    assert record['passed'] and record['actual_full_pretraining_epochs_executed']==0
    (tmp_path/'native_check.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
