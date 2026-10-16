from datetime import datetime,timezone
from pathlib import Path
import csv
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import publish_TAB_complete_case_catalog_v60 as p
ROOT=p.ROOT;c=p.read(ROOT/p.CONFIG);out=ROOT/c['output_directory']
q=p.read(ROOT/c['original_TAB_configuration']);assert len(q['entries'])==878
assert sum(len(e['case']['required_label_policies']) for e in q['entries'])==2057
original_base=ROOT/q['state_root']
state=p.read(ROOT/'experiments/runs/fm_TAB878_fair_entity_continuation_v2/status.json')
recovery=p.read(ROOT/'experiments/runs/fm_TAB878_resource_durable_recovery_v2/status.json')
for actor in [state,recovery]:
    proc=psutil.Process(actor['pid']);assert proc.is_running() and proc.create_time()==actor['create_time']
complete=set(state['completed_case_keys']);rows=[]
for entry in q['entries']:
    key=entry['key'];case=entry['case'];done=key in complete
    if done:
        summary_path=Path(q['artifact_root'])/'cases'/key/'summary/complete_case_TAB_metric_controls.json'
        guard_path=original_base/'attempts'/key/'summary/guard_receipt.json'
        report,guard=p.read(summary_path),p.read(guard_path)
        assert report['passed'] and report['case']==case and report['entities']=={'smap':54,'msl':27,'smd':28}[case['dataset']]
        assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['proof_sha256']==p.sha(summary_path)
    failed=key in state['failed_cases_preserved']
    rows.append(dict(case_key=key,case_id=case['id'],family=case['family'],dataset=case['dataset'],seed=case['seed'],track=entry['track'],
        required_label_views=len(case['required_label_policies']),required_entities={'smap':54,'msl':27,'smd':28}[case['dataset']],
        original_metric_case_complete=done,source_case_imported_into_v60=key=='e8a311b6a8a4968fb22a',
        original_failed_case_preserved=failed,original_complete_entities=state['completed_entities'].get(key,0),
        recovered_entities=recovery['completed_recovered_entities'].get(key,0),
        recovery_reused_original_entities=recovery['reused_original_entities'].get(key,0),
        status='complete_original_metric_control_case' if done else 'original_failure_preserved_resource_recovery_pending_or_partial' if failed else 'not_yet_complete_in_this_TAB_registry'))
with (out/'all878_original_TAB_case_scope_and_captured_status.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
shutil.copyfile(ROOT/'experiments/runs/fm_TAB878_fair_entity_continuation_v2/status.json',out/'captured_original_TAB_continuation_state.json')
shutil.copyfile(ROOT/'experiments/runs/fm_TAB878_resource_durable_recovery_v2/status.json',out/'captured_preserved_resource_recovery_state.json')
test_path=Path('F:/aicoding/IIA_Data/experiments/TAB_complete_case_catalog_v60_20261016_final_tests.xml')
suites=list(ET.parse(test_path).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==9
assert all(int(x.get('failures',0))==int(x.get('errors',0))==int(x.get('skipped',0))==0 for x in suites)
shutil.copyfile(test_path,out/'relevant_validation.xml')
with (out/'README.md').open('a',encoding='utf-8') as f:
    f.write('\n随附全部878案例的独立执行范围矩阵，以及原失败与恢复状态的捕获快照；只读核验已完成案例的实际汇总守卫。矩阵保留2057个所需标签视图，不以当前804条指标缩小原始范围。9项新目录语义检查通过，覆盖缺失实体、部分均值冒充完整值、错误实体区间及虚假认证的拒绝规则。\n')
shutil.copyfile(__file__,out/'supplement_source.py')
(out/'packet_integrity.json').write_text(json.dumps(dict(files=[dict(path=x.relative_to(out).as_posix(),sha256=p.sha(x)) for x in sorted(out.rglob('*')) if x.is_file() and x.name!='packet_integrity.json']),indent=2)+'\n',encoding='utf-8')
ara=ROOT/'projects/flow_matching_research/ara/execution/TAB_complete_case_catalog.2026-10-16_v60.json'
assert not ara.exists()
ara.write_text(json.dumps(dict(result_packet=out.relative_to(ROOT).as_posix(),metric_rows=11795,numeric_rows=3084,
    appended_complete_fixed_model_cases=1,appended_full54_finite_macro_metrics=700,appended_undefined_full_macro_metrics=104,
    original_TAB_scope_cases=878,original_TAB_scope_label_views=2057,all_original_experiments_complete=False),indent=2)+'\n',encoding='utf-8')
paths=[p.CONFIG,p.TEST,p.MODULE.replace('.','/')+'.py',ara.relative_to(ROOT).as_posix(),'projects/flow_matching_research/README.rst']
paths += [x.relative_to(ROOT).as_posix() for x in out.rglob('*') if x.is_file()]
head=lambda path:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD']).decode().strip()
target=ROOT/'tmp/TAB_complete_case_catalog_v60_manifest_20261016_v1.json';assert not target.exists()
target.write_text(json.dumps(dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(files=len(paths),metric_rows=11795,numeric_rows=3084,complete_cases_in_scope=len(complete),required_TAB_cases=878,required_label_views=2057)))
