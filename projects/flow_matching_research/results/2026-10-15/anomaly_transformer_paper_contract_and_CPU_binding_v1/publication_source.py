"""Publish complete runnable registrations, actual idle handoff and exact tests."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil

from scripts.flow_matching.run_anomaly_transformer_author_full_v1 import ROOT,read,sha,write
from scripts.flow_matching.run_anomaly_transformer_paper_contract_v1 import verify as verify_paper
from scripts.flow_matching.run_anomaly_transformer_explicit_cpu_binding_v3 import verify as verify_cpu

paperpath='configs/experiments/fm_anomaly_transformer_paper_contract.v1.json'
cpupath='configs/runtime/fm_anomaly_transformer_explicit_cpu_binding.v3.json'
paper,config=read(ROOT/paperpath),read(ROOT/cpupath);verify_paper(paper);verify_cpu(config)
xmls=[('paper_contract_validation.xml',ROOT/'tmp/AnomalyTransformer_paper_contract_validation_20261015_v1.xml',19),
      ('explicit_CPU_binding_validation.xml',ROOT/'tmp/AnomalyTransformer_explicit_cpu_binding_validation_20261015_v3.xml',5)]
for _,path,count in xmls:
    suites=list(ET.parse(path).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==count
    assert all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
handoff=read(ROOT/config['state_root']/'idle_handoff_receipt.json')
assert handoff['numerical_workers_stopped']==0 and len(handoff['records'])==2
live=[]
for record in handoff['records']:
    p=psutil.Process(record['new_pid']);assert abs(p.create_time()-record['new_create_time'])<.001
    assert 'scripts.flow_matching.run_anomaly_transformer_explicit_cpu_binding_v3' in p.cmdline()
    branch=record['branch'];original=read(ROOT/config['cases'][branch]);state=read(ROOT/original['state_root']/'status.json')
    assert state['pid']==p.pid
    live.append(dict(branch=branch,pid=p.pid,create_time=p.create_time(),command=p.cmdline(),state_at_capture=state))
p=psutil.Process(8528);assert abs(p.create_time()-1792041972.191672)<.001
assert 'scripts.flow_matching.run_anomaly_transformer_paper_contract_v1' in p.cmdline()
state=read(ROOT/paper['state_root']/'status.json');assert state['pid']==p.pid
live.append(dict(branch='primary_paper_reconstruction',pid=p.pid,create_time=p.create_time(),command=p.cmdline(),state_at_capture=state))
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/anomaly_transformer_paper_contract_and_CPU_binding_v1'
folder.mkdir(exist_ok=False)
write(folder/'actual_live_controller_receipt.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    controllers=live,new_main_paper_reconstruction_jobs=25,score_streams_per_model_split=7,
    independent_validation=True,original_author_matrices_and_guards_preserved=True,
    benchmark_models_promoted_by_this_registration=0,all_original_experiments_complete=False))
sources={
 'frozen_paper_registration.json':ROOT/paperpath,
 'paper_runtime_source.py':ROOT/'scripts/flow_matching/run_anomaly_transformer_paper_contract_v1.py',
 'paper_registration_source.py':ROOT/'scripts/flow_matching/register_anomaly_transformer_paper_contract_v1.py',
 'paper_model_config.json':ROOT/'configs/models/fm_anomaly_transformer_paper_contract.v1.json',
 'paper_contract_tests.py':ROOT/'tests/test_fm_anomaly_transformer_paper_contract_v1.py',
 'frozen_CPU_binding_v3.json':ROOT/cpupath,
 'preserved_unlaunched_CPU_binding_v2.json':ROOT/'configs/runtime/fm_anomaly_transformer_explicit_cpu_binding.v2.json',
 'CPU_binding_base_source_v2.py':ROOT/'scripts/flow_matching/run_anomaly_transformer_explicit_cpu_binding_v2.py',
 'CPU_binding_complete_interface_v3.py':ROOT/'scripts/flow_matching/run_anomaly_transformer_explicit_cpu_binding_v3.py',
 'CPU_binding_tests_v3.py':ROOT/'tests/test_fm_anomaly_transformer_explicit_cpu_binding_v3.py',
 'actual_idle_handoff_receipt.json':ROOT/config['state_root']/'idle_handoff_receipt.json',
 'idle_handoff_source.py':ROOT/'tmp/handoff_anomaly_transformer_explicit_cpu_idle_20261015_v3.py'}
for branch in ['author','restored']:
    sources[f'{branch}_runtime_binding.json']=ROOT/config['state_root']/branch/'binding.json'
for name,path in sources.items():shutil.copyfile(path,folder/name)
for name,path,_ in xmls:shutil.copyfile(path,folder/name)
(folder/'publication_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
(folder/'README.md').write_text('Primary-paper reconstruction and verified Windows CPU binding\n\n'
 'All25 five-real-dataset/five-seed main slots are now registered as a separate primary-paper-contract '
 'reconstruction. It implements Table13 contiguous80/20train/validation counts,train-only normalization, '
 '512-width three-layer author model,batch32,non-overlapped100windows,max10epochs,Algorithm2head means '
 'before symmetricKL and layer mean,Algorithm3feature mean and temperature1. Both detached Equation5 '
 'losses use per-window Frobenius/L1 sums and one accumulated Adam update. Test data select neither '
 'checkpoint nor threshold. All seven criterion/layer streams and components are saved over full covered '
 'train/validation/test splits; ratio/fixed-delta controls and seven ROC operating points are reported '
 'separately from continuous scoreAUROC. Original omitted tails remain explicit.\n\n'
 'This is an explicit reconstruction, not a claim that unspecified choices are paper-authenticated: '
 'patience3,two-loss early stopping,learning-rate schedule,FFN width,Gaussian positivity transformation, '
 'log epsilon,phase-update convention and exact historical split identity remain disclosed. The sixth '
 'NeurIPS-TS benchmark,training ablations/sensitivity,adjacent-association studies,baselines,independent '
 'replay,uncertainty and strict TAB remain required in the unchanged45/681/273objective.\n\n'
 'A real isolated fixture exposed that empty CUDA_VISIBLE_DEVICES did not reliably disable CUDA on '
 'this Windows runtime. Explicit-1was checked against actual torch.cuda.is_available()==False and '
 'actual full-width source Solver parameter devices for both author and SWaT-restored branches. '
 'Only exact idle metadata owners26492/30512 were suspended,rechecked for waiting/no-active-worker '
 'and console-only children,then retired. New exact PID/create-time bindings are archived. Zero '
 'numerical workers were stopped. Frozen old source/model/config bytes,budgets,all40GiBcommit/32GiBphysical '
 'admission and16GiBemergency guards remain. Version2binding draft was never launched; prelaunch '
 'inspection found its subprocess proxy lacked STDOUT,so version3preserves the complete API.\n\n'
 'Nineteen paper reconstruction checks plus five explicit-CPU/interface checks passed. Small fixture '
 'updates verify plumbing and mathematical invariants only; zero benchmark completions or performance '
 'claims are added by this registration. Three controllers were actually live when captured.\n',encoding='utf-8')
files={p.relative_to(folder).as_posix():sha(p) for p in folder.iterdir() if p.is_file()}
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    relevant_tests_passed=24,paper_checks=19,CPU_interface_checks=5,artifact_files_sha256=files,
    numerical_workers_stopped=0,additional_benchmark_completions_claimed=0,all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(),paperpath,cpupath,
 'configs/runtime/fm_anomaly_transformer_explicit_cpu_binding.v2.json',
 'configs/models/fm_anomaly_transformer_paper_contract.v1.json',
 'scripts/flow_matching/run_anomaly_transformer_paper_contract_v1.py',
 'scripts/flow_matching/register_anomaly_transformer_paper_contract_v1.py',
 'scripts/flow_matching/run_anomaly_transformer_explicit_cpu_binding_v2.py',
 'scripts/flow_matching/run_anomaly_transformer_explicit_cpu_binding_v3.py',
 'tests/test_fm_anomaly_transformer_paper_contract_v1.py',
 'tests/test_fm_anomaly_transformer_explicit_cpu_binding_v2.py',
 'tests/test_fm_anomaly_transformer_explicit_cpu_binding_v3.py']
paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
manifest=dict(previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths))
target=ROOT/'tmp/AnomalyTransformer_paper_and_CPU_binding_publication_manifest_20261015_v1.json'
target.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(exact_files=len(paths),tests_passed=24,live_controllers=[r['pid'] for r in live],
                      full_main_jobs_registered=25,full_score_streams=7,numerical_workers_stopped=0)))
