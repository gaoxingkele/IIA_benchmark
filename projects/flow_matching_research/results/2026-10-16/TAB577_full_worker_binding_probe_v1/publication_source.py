from pathlib import Path
import hashlib,json,shutil,subprocess
from datetime import datetime,timezone
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import run_common105_matched577_TAB_v1 as s

root=Path.cwd();out=root/'projects/flow_matching_research/results/2026-10-16/TAB577_full_worker_binding_probe_v1'
if out.exists():
    assert {p.name for p in out.iterdir()}<={'validation4.xml','test_source.py','synthetic_complete_worker_proofs_not_benchmark.json'}
else:out.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
write=lambda name,v:(out/name).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
test='tests/test_fm_common105_TAB577_full_worker_binding_v1.py'
base=Path('F:/aicoding/IIA_Data/experiments')
xml=base/'TAB577_worker_probe_v2.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(t.get('tests',0)) for t in suites)==4
assert not sum(int(t.get('failures',0))+int(t.get('errors',0)) for t in suites)
shutil.copy2(xml,out/'validation4.xml');shutil.copy2(root/test,out/'test_source.py')
reports=[]
for p in (base/'TAB577_worker_probe_v2').rglob(s.REPORT):
    proof=read(p);assert proof['passed']
    reports.append(dict(path=str(p),sha256=sha(p),case_id=proof['case']['id'],fixture_only=True,
        views=len(proof['metric_results']),calls=proof['metric_calls'],outcomes=proof['metric_outcome_counts'],
        cells=proof['durable_cell_journal']['fresh_cells'],
        original_component_sha256=proof['original_component_sha256'],no_scientific_guard_created=True))
assert sorted(p['calls'] for p in reports)==[402,804]
write('synthetic_complete_worker_proofs_not_benchmark.json',reports)
state=read(root/'experiments/runs/fm_TAB_full878_original_metric_suite_v1/status.json')
live=[]
for pid in [state['pid'],state['active_pid'],36912,40268,41992]:
    if pid is None:continue
    try:
        p=psutil.Process(pid)
        live.append(dict(pid=pid,create_time=p.create_time(),command=p.cmdline(),cpu_seconds=sum(p.cpu_times()[:2]),alive=p.is_running()))
    except psutil.NoSuchProcess:
        live.append(dict(pid=pid,alive=False,interpretation='Observed child already exited; re-read current owner state, do not restart.'))
write('actual_existing_processes_and_original_TAB_progress.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),processes=live,original_TAB=state,
    owner_state_reread=read(root/'experiments/runs/fm_TAB_full878_original_metric_suite_v1/status.json'),full_research_complete=False))
write('validation_and_preserved_initial_test_failure.json',dict(passed_tests=4,
    test_source=dict(path=test,sha256=sha(root/test)),passing_xml=dict(path=str(xml),sha256=sha(xml)),
    initial_failure=dict(path=str(base/'TAB577_worker_probe_v1.xml'),sha256=sha(base/'TAB577_worker_probe_v1.xml'),
        tests=4,failed=2,passed=2,initial_source_path=str(base/'TAB577_worker_probe_initial_test_v1.py'),
        initial_source_sha256=sha(base/'TAB577_worker_probe_initial_test_v1.py'),
        reason='Test mistakenly required separate execution elapsed_seconds to be equal. Corrected only wall-clock comparison; all numerical values/status/warnings/errors/stdout remain exact; no production code or tolerances changed.'),
    coverage='Actual new full-worker adapter with SMAP4 and MSL2 overlays:1206 original calls plus1206 direct comparisons; missing closure rejected; changed label bytes rejected; no guard means no scientific result certification.',
    production_definition_unchanged=True,synthetic_not_benchmark_or_CUDA=True,
    full45paper681method273data702scalar_scope_remains=True))
(out/'README.md').write_text('''# 新公共切分 TAB 完整工作流的补充验证

新增 4 项测试通过。使用真实原始 TAB 指标函数，在合成 40 点输入上通过新接口完整执行 SMAP 的四组标签口径（804 项调用）及 MSL 的两组标签口径（402 项调用），并与直接调用原完整指标程序逐项核对指标值、状态、警告、异常和原始输出。保存的逐指标记录数与全部调用一致。运行耗时分别保留，不要求两次计算耗时相同。

另验证没有原生闭合证据时不能开始、标签字节变化时在计算前拒绝，以及没有实际资源守卫时完整合成输出仍不能认证为科学结果。测试仅在独立合成目录替换就绪提供者，未伪造生产守卫，没有训练或推理模型，没有改变生产代码、数值定义、阈值、模型预算或资源策略。

首次测试中，两个完整流程用例因测试错误地把执行耗时纳入全等比较而失败；失败 XML 和初版测试文件保留在 F 盘，并记录路径与 SHA256。修正测试后改用新输出目录，4 项全部通过。其余所有结果字段仍要求相等。

此前的 43 项验证和本次 4 项验证覆盖不同部分，共 47 项。合成指标值不作为数据集成绩。实际 TAB、CFM、BRITS 和 CrossAD 进程及命令单独记录；原始 TAB 实体进度并不表示完整数据集或全部研究已完成。全部 45 篇论文、681 项方法/基线/消融、273 项数据任务、702 项论文标量目标继续保留。
''',encoding='utf-8',newline='\n')
shutil.copy2(__file__,out/'publication_source.py')
git=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
paths=[test]+[p.relative_to(root).as_posix() for p in out.iterdir() if p.is_file()]
(root/'tmp/TAB577_worker_probe_exact_publication_v1.json').write_text(json.dumps(dict(previous_root=git(root),previous_parent=git(root.parent/'flow-matching-research'),paths=paths),indent=2),encoding='utf-8')
print(json.dumps(dict(tests=4,metric_calls=1206,packet=str(out),files=len(paths))))
