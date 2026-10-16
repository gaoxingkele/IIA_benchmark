"""Fill only fifteen pending controls of the actual22nd complete condition."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import publish_crossad_row5_native_increment_v67 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

packet=ROOT/s.PACKET
for pin in read(packet/'packet_integrity.json')['files']:assert sha(packet/pin['path'])==pin['sha256']
assert read(packet/'iteration_status.json')['CrossAD_full_CUDA_conditions']==22
assert not (ROOT/s.CONFIG).exists()
with (ROOT/s.CONFIG).open('x',encoding='utf-8') as f:json.dump(s.registration(),f,indent=2)
status=s.publish();out=ROOT/read(ROOT/s.CONFIG)['output_directory']
with (ROOT/s.BASE/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:before=list(csv.DictReader(f))
with (out/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:after=list(csv.DictReader(f))
updates=[b for a,b in zip(before,after) if a!=b]
assert len(updates)==15 and s.merge(before,updates)==after and sum(bool(r['mean']) for r in after)==3205
assert all(a==b for a,b in zip(before,after) if s.IDENTITY not in a['protocol'])
for name in ['all878_original_TAB_case_scope_and_captured_status.csv','five_shared_original_SMD_comparison_references.csv']:
    assert sha(ROOT/s.BASE/name)==sha(out/name)
xml=ROOT/'tmp/catalog_v67_validation_20261016_v1.xml';suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==8
assert all(int(x.get(k,0))==0 for x in suites for k in ['failures','errors','skipped'])
shutil.copyfile(xml,out/'relevant_fresh8_validation.xml');shutil.copyfile(ROOT/s.CONFIG,out/'frozen_publication_configuration.json')
shutil.copyfile(__file__,out/'packaging_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
page=ROOT/'projects/flow_matching_research/README.rst';text=page.read_text(encoding='utf-8')
head,separator,body=text.partition('原论文单细胞轨道另见')
assert separator
for a,b in [
    ('2026-10-16 的 v66 增量快照','2026-10-16 的 v67 增量快照'),
    ('all_algorithm_dataset_metrics_v66/','all_algorithm_dataset_metrics_v67/'),
    ('共11925行指标，3190行有数值','共11925行指标，3205行有数值'),
    ('原v65的11921行全部字段和捕获时间保持；仅填入Spectral第9条件的4项原待完成指标。','原v66的11910行全部字段和捕获时间保持；仅填入CrossAD第22条件的15项原待完成指标。'),
    ('该快照已核验分别21和9个条件','该快照已核验分别22和9个条件')]:
    assert a in head,a;head=head.replace(a,b)
text=head+separator+body
text+='''
`完整算法—数据集指标表v67 <results/2026-10-16/all_algorithm_dataset_metrics_v67/README.md>`_
11925行全部保留，3205行有数值；仅填入CrossAD row5/seed1105的15项原待完成指标，其他11910行与捕获时间保持。
CrossAD22/139、五种重建消融各3/5、完整主要组仍1/22；Spectral9/70、完整五模型组1/14，8项增量保护验证通过。
工业466个原条件及FM25同分数PA控制仍单列原始证据轨道，原TAB878案例和全部论文/方法/数据/标量范围保持，整体未完成。
'''
page.write_text(text,encoding='utf-8',newline='\n')
ara=ROOT/'projects/flow_matching_research/ara/execution/CrossAD_row5_catalog.2026-10-16_v67.json'
assert not ara.exists();write(ara,dict(packet=out.relative_to(ROOT).as_posix(),metric_rows=11925,numeric_rows=3205,
    unchanged_rows_all_fields=11910,new_complete_fixed_case_controls=15,all_original_experiments_complete=False))
manifest=read(ROOT/'tmp/CrossAD_row5_seed1105_full22_publication_manifest_20261016_v1.json')
gethead=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
assert gethead(ROOT)==manifest['previous_root'] and gethead(ROOT.parent/'flow-matching-research')==manifest['previous_parent']
paths=set(manifest['paths'])|{s.MODULE.replace('.','/')+'.py',s.TEST,s.CONFIG,page.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix()}
paths|={p.relative_to(ROOT).as_posix() for p in out.rglob('*') if p.is_file()}
write(ROOT/'tmp/CrossAD22_and_catalog_v67_publication_manifest_20261016_v1.json',dict(previous_root=gethead(ROOT),previous_parent=gethead(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(publication_files=len(paths),CrossAD_full_conditions=22,metric_rows=11925,numeric_rows=3205,changed_rows=15)))
