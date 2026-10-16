"""Publish the ninth real generation replay and four complete pending controls."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import publish_spectral_energy45_native_increment_v66 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

packet=ROOT/s.PACKET
for pin in read(packet/'packet_integrity.json')['files']:assert sha(packet/pin['path'])==pin['sha256']
assert read(packet/'iteration_status.json')['full_generation_evaluator_reexecutions']==9
page=packet/'README.md'
page.write_text(page.read_text(encoding='utf-8')+'\n新减原差异同时包含新采样与重新训练/执行评估器的随机性，不能把差值完全归因于采样或生成模型能力变化。\n',encoding='utf-8',newline='\n')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file() and p.name!='packet_integrity.json']))
assert not (ROOT/s.CONFIG).exists()
with (ROOT/s.CONFIG).open('x',encoding='utf-8') as f:json.dump(s.registration(),f,indent=2)
state=s.publish();out=ROOT/read(ROOT/s.CONFIG)['output_directory']
with (ROOT/s.BASE/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:before=list(csv.DictReader(f))
with (out/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:after=list(csv.DictReader(f))
updates=[b for a,b in zip(before,after) if a!=b]
assert len(updates)==4 and s.merge(before,updates)==after and sum(bool(r['mean']) for r in after)==3190
for name in ['all878_original_TAB_case_scope_and_captured_status.csv','five_shared_original_SMD_comparison_references.csv']:
    assert sha(ROOT/s.BASE/name)==sha(out/name)
xml=ROOT/'tmp/catalog_v66_validation_20261016_v1.xml';suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==8
assert all(int(x.get(k,0))==0 for x in suites for k in ['failures','errors','skipped'])
shutil.copyfile(xml,out/'relevant_fresh8_validation.xml');shutil.copyfile(ROOT/s.CONFIG,out/'frozen_publication_configuration.json')
shutil.copyfile(__file__,out/'packaging_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
page=ROOT/'projects/flow_matching_research/README.rst';text=page.read_text(encoding='utf-8')
for a,b in [
    ('2026-10-16 的 v65 增量快照','2026-10-16 的 v66 增量快照'),
    ('all_algorithm_dataset_metrics_v65/','all_algorithm_dataset_metrics_v66/'),
    ('共11925行指标，3186行有数值','共11925行指标，3190行有数值'),
    ('原v64的11910行全部字段和捕获时间保持；仅填入CrossAD第21条件的15项原待完成指标。','原v65的11921行全部字段和捕获时间保持；仅填入Spectral第9条件的4项原待完成指标。'),
    ('该快照已核验分别21和8个条件','该快照已核验分别21和9个条件')]:
    assert a in text,a;text=text.replace(a,b)
text+='''
`完整指标表v66：新增Spectral Energy45的4项完整指标 <results/2026-10-16/all_algorithm_dataset_metrics_v66/README.md>`_
11925行全部保留，3190行有数值，其他11921行全部字段和捕获时间保持；8项增量保护验证通过。
生成重跑9/70、完整五训练模型组1/14；Energy为4/5，不以内部五次评估重复替代训练种子统计。
CrossAD21/139、原TAB878案例、原SPOT25及全部论文/方法/数据/标量范围保持，整体未完成。
'''
page.write_text(text,encoding='utf-8',newline='\n')
ara=ROOT/'projects/flow_matching_research/ara/execution/Spectral_Energy45_catalog.2026-10-16_v66.json'
assert not ara.exists();write(ara,dict(packet=out.relative_to(ROOT).as_posix(),metric_rows=11925,numeric_rows=3190,
    unchanged_rows_all_fields=11921,new_complete_fixed_case_controls=4,all_original_experiments_complete=False))
manifest=read(ROOT/'tmp/Spectral_Energy45_full9_manifest_20261016_v1.json')
head=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
assert head(ROOT)==manifest['previous_root'] and head(ROOT.parent/'flow-matching-research')==manifest['previous_parent']
paths=set(manifest['paths'])|{s.MODULE.replace('.','/')+'.py',s.TEST,s.CONFIG,page.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix()}
paths|={p.relative_to(ROOT).as_posix() for folder in [out,packet] for p in folder.rglob('*') if p.is_file()}
write(ROOT/'tmp/Spectral9_and_catalog_v66_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(publication_files=len(paths),complete_generation_conditions=9,metric_rows=11925,numeric_rows=3190,changed_rows=4)))
