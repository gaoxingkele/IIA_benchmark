"""Launch the new remaining-seed audit owner once; archive bounded evidence."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import run_timesnet_SMAP_remaining_full_audits_v2 as s

ROOT,read,write,sha=s.ROOT,s.read,s.write,s.sha
c=read(ROOT/s.CONFIG);jobs=s.verify(c)
validation=ROOT/'tmp/TimesNet_remaining4_full_audits_validation_20261016_v2.xml'
suites=list(ET.parse(validation).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==50
assert all(int(x.get(k,0))==0 for x in suites for k in ['failures','errors','skipped'])
folder=ROOT/'projects/flow_matching_research/results/2026-10-16/TimesNet_SMAP_remaining4_full_audits_registered_v2'
assert not folder.exists()
base=ROOT/c['state_root'];assert not base.exists()
old_evidence={}
for mode,path in c['existing_seed1103_configuration'].items():
    old=read(ROOT/path);s.original.binding(old)
    proof=Path(old['output_directory'])/'independent_full_saved_score_verification.json'
    p=read(ROOT/proof);guard_path=ROOT/old['state_root']/'receipt.json';g=read(guard_path)
    assert p['passed'] and p['configuration_sha256']==sha(ROOT/path)
    assert p['mode']==mode and p['full_entities']==51 and p['full_test_points']==408160
    assert g['exit_code']==0 and not g['resource_guard_aborted'] and g['full_saved_score_review_passed']
    if mode=='native':
        assert p['full_native_checkpoint_reinference_passed'] and p['all_replayed_timestamp_scores_bitwise_equal']
        assert sha(ROOT/old['output_directory']/'replayed_scores.npz')==p['replay_scores_sha256']
    old_evidence[mode]=dict(configuration=path,proof_path=str(ROOT/proof),proof_sha256=sha(ROOT/proof),
        guard_path=str(guard_path),guard_sha256=sha(guard_path))
for process in psutil.process_iter(['cmdline']):assert s.MODULE not in (process.info['cmdline'] or [])
base.mkdir(parents=True,exist_ok=False)
command=[sys.executable,'-B','-X','utf8','-u','-m',s.MODULE]
with (base/'controller.stdout.log').open('xb') as out,(base/'controller.stderr.log').open('xb') as err:
    child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src')),
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command,configuration_sha256=sha(ROOT/s.CONFIG))
write(base/'launch.json',handle)
for _ in range(70):
    assert child.poll() is None,'Preserve actual owner failure and logs;never automatically restart'
    if (base/'status.json').exists():break
    time.sleep(.1)
status=read(base/'status.json')
assert status['pid']==handle['pid'] and status['create_time']==handle['create_time']
assert psutil.Process(child.pid).cmdline()==command
folder.mkdir(parents=True,exist_ok=False)
sources=dict(frozen_registration=ROOT/s.CONFIG,controller_launch=base/'launch.json',status_at_registration=base/'status.json',
    original_source=ROOT/(s.MODULE.replace('.','/')+'.py'),original_tests=ROOT/s.TEST,validation_50_checks=validation)
job=jobs[s.identity(1104)];result=read(ROOT/job['output_directory']/'result.json')
sources['seed1104_original_completed_training_result']=ROOT/job['output_directory']/'result.json'
sources['seed1104_original_guarded_training_receipt']=ROOT/s.stage_config(c,job,'point')['terminal_receipt']
for mode,record in old_evidence.items():
    sources['seed1103_existing_'+mode+'_proof']=Path(record['proof_path'])
    sources['seed1103_existing_'+mode+'_guard']=Path(record['guard_path'])
for name,path in sources.items():
    dest=folder/(name+path.suffix);shutil.copyfile(path,dest);assert sha(path)==sha(dest)
write(folder/'existing_seed1103_current_full_proof_validation.json',old_evidence)
summary=dict(captured_utc=datetime.now(timezone.utc).isoformat(),remaining_original_seeds=list(s.SEEDS),full_audit_stages=8,
    original_required_cohort=[1103,*s.SEEDS],already_closed_seed1103_revalidated=True,controller=handle,
    original_seed1104_guarded_training_completed=True,original_seed1104_full_training_seconds=result['training_seconds'],
    seed1104_source_metrics_pending_new_independent_review=result['metrics']['strict']['1']['micro'],
    original51_entity_subset_preserved=True,canonical_NASA_full_dataset_equivalence_certified=False,
    original_numerical_worker_bytecode_unchanged=True,original_resource_policy_unchanged=True,tests_passed=50,
    remaining_full_audits_complete=False,all_original_experiments_complete=False)
write(folder/'registration_summary.json',summary)
(folder/'README.md').write_text('''# TimesNet SMAP：剩余四个原种子的完整审计已登记

原 seed1103 的分数独立复核与全模型重推理已经成功，本次重新核对原配置、完整模型/数据/分数及两阶段成功证明，并核验重推理分数文件当前哈希，不重复推理 seed1103。

新增队列覆盖原登记 seed1104、1105、1106、1107，每个种子分别执行保存分数独立复核和完整检查点重推理，共 8 个阶段。seed1104 已自然完成原定训练和全部评分，资源回执 exit0、无中止；本队列等待其余各原模型自然完成后依次复核。原训练配置为 3 轮，已有预算保留回执，但没有独立保存的逐 epoch/更新计数，因此不将配置轮数冒充实际更新轨迹。

复用 seed1103 原审计器完全相同的数值 worker 字节码，私有全局字典仅替换真实种子绑定函数。原模板默认 seed1103 保留，实际模型构建和检查点使用各任务自己的 seed override。元数据验证先检查真实 ID、种子、覆盖、原指纹，再在私有副本中适配旧验证器的固定选择器；原配置、结果、模块和数值 worker 完全不修改。50 项测试通过，覆盖完整种子身份、容量/轮数/窗口/批量/资源拒绝、共享全局不变、旧数值实现及冷启动门槛。

原模型 28,132,633 参数、CPU float32、批量 128、2 线程、原内存和提交量上限、原只读锁全部保留。每项检查所有 51 个本地登记 SMAP 实体、全部 TRAIN/VAL/TEST 时间戳，重推理分数要求原环境下逐字节一致；独立核对仅验证集定阈值的点指标，以及原事件、bootstrap 和测试校准诊断指标。测试标签选择阈值和 PA 诊断不混作严格指标。数据使用原登记子集，未认证与 NASA 原文全量数据完全对应。

本包只是训练完成证据及审计登记，不宣称剩余 8 个阶段已经通过。所有大型原检查点、数据和新分数保留 F 盘，失败与部分尝试保留且不自动重试。完整 TAB 范围指标、五种子结果归约、其他 35 个基线任务、291 个预处理对照和所有其他论文实验仍待相应证据逐项完成。

来源：[TimesNet](https://arxiv.org/abs/2210.02186)、[TAB](https://github.com/decisionintelligence/TAB)。
''',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(tests_passed=50,
    files_sha256={p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file()},all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/TimesNet_SMAP_remaining4_full_audits.2026-10-16_v2.json'
write(ara,dict(original_remaining_seeds=list(s.SEEDS),registered_full_stages=8,evidence=folder.relative_to(ROOT).as_posix(),
    full_remaining_audits_complete=False,source_worker_bytecode_unchanged=True,all_original_experiments_complete=False))
paths=[s.CONFIG,s.TEST,s.MODULE.replace('.','/')+'.py',ara.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()]
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/TimesNet_remaining4_full_audits_publication_manifest_20261016_v2.json',
    dict(previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(controller=handle,initial_state=status,registered_full_stages=8,tests_passed=50,exact_files=len(paths))))
