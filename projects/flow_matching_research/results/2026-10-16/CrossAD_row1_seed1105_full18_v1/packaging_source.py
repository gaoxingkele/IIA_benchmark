from datetime import datetime,timezone
from pathlib import Path
import csv
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import collect_crossad_full139_cold_handoff_v1 as cold
from scripts.flow_matching.audit_crossad_full_native_completion_v1 import training_budget
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

packet=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_row1_seed1105_full18_v1'
assert read(packet/'iteration_status.json')['CrossAD_full_CUDA_conditions']==18
_,q=cold.verify(read(ROOT/cold.CONFIG));budgets={}
with (packet/'same_seed_row1_vs_full_three_seeds.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
assert len(rows)==9 and {int(r['training_seed']) for r in rows}=={1103,1104,1105}
for r in rows:
    seed=int(r['training_seed'])
    for kind,cid in [('row1',f'crossad_ablation__row1__gecco__seed{seed}'),('full_model',f'crossad_author__fresh_train__gecco__seed{seed}')]:
        job=next(j for j in q['jobs'] if j['canonical_CPU_condition_reference']==cid)
        actual=read(ROOT/q['delegated_first_configuration'])['job'] if cid==q['delegated_first_original_condition'] else job
        path=ROOT/actual['output_directory']/'result.json';result=read(path)
        proof=read(read(packet/'all18_proof_bindings.json')['records'][cid]['full_array_proof_path'])
        assert sha(path)==proof['source_result_sha256']
        budget=training_budget(actual,result['details'],read(path.parent/'training_trace.json'))
        assert budget['original_maximum_epochs']==20 and budget['original_patience']==3
        budgets[cid]=dict(actual_original_budget=budget,result_sha256=sha(path),training_trace_sha256=sha(path.parent/'training_trace.json'))
        r[kind+'_actual_epochs']=budget['actual_completed_epochs'];r[kind+'_actual_updates']=budget['full_training_forward_calls']
    assert abs(float(r['row1_minus_full_F1'])-(float(r['row1_F1'])-float(r['full_F1'])))<1e-14
assert len(budgets)==6
with (packet/'all9_same_seed_comparisons_with_actual_training_budget.csv').open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write(packet/'all6_paired_original_training_budget_bindings.json',dict(records=budgets,
    partial_seed_mean_or_CI_computed=False,actual_update_counts_may_differ_under_same_original_stopping_rule=True))
xml=Path('F:/aicoding/IIA_Data/experiments/CrossAD_full18_row1_three_seeds_20261016_v1_tests.xml')
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==15
assert all(int(x.get('errors',0))==int(x.get('failures',0))==int(x.get('skipped',0))==0 for x in suites)
shutil.copyfile(xml,packet/'relevant_fresh15_full18_validation.xml')
shutil.copyfile(ROOT/'tmp/close_CrossAD_row1_seed1105_full18_20261016_v1.py',packet/'closure_source.py')
page=packet/'README.md';text=page.read_text(encoding='utf-8')
old='本轮还将重新执行相关15项检查'
assert text.count(old)==1
text=text.replace(old,'本轮重新执行相关15项检查全部通过，附新的实际XML')
page.write_text(text+'\n另附三组同种子、三个验证阈值的9行比较及全部6个参与模型的实际训练预算。保留早停导致的实际更新量差异；不计算3/5种子的部分均值或区间，不将其解释成完全匹配计算量的纯组件因果效应。\n',encoding='utf-8')
shutil.copyfile(__file__,packet/'packaging_source.py')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file() and p.name!='packet_integrity.json']))
ara='projects/flow_matching_research/ara/execution/CrossAD_row1_seed1105_full18.2026-10-16_v1.json'
paths=[ara,'projects/flow_matching_research/README.rst']+[p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
head=lambda path:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD']).decode().strip()
manifest=ROOT/'tmp/CrossAD18_row1_three_seeds_manifest_20261016_v1.json';assert not manifest.exists()
write(manifest,dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(files=len(paths),full_CrossAD_conditions=18,row1_primary_seeds_complete=3,paired_point_controls=9)))
