"""Append only the actual full21 condition to the unchanged catalog scope."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import publish_crossad_row4_native_increment_v65 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

packet=ROOT/s.PACKET
for pin in read(packet/'packet_integrity.json')['files']:assert sha(packet/pin['path'])==pin['sha256']
assert read(packet/'iteration_status.json')['CrossAD_full_CUDA_conditions']==21
assert not (ROOT/s.CONFIG).exists()
with (ROOT/s.CONFIG).open('x',encoding='utf-8') as f:json.dump(s.registration(),f,indent=2)
state=s.publish();out=ROOT/read(ROOT/s.CONFIG)['output_directory']
with (ROOT/s.BASE/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:before=list(csv.DictReader(f))
with (out/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:after=list(csv.DictReader(f))
updates=[b for a,b in zip(before,after) if a!=b]
assert len(updates)==15 and s.merge(before,updates)==after
assert sum(bool(r['mean']) for r in after)==3186 and len(after)==11925
assert all(a==b for a,b in zip(before,after) if s.IDENTITY not in a['protocol'])
for name in ['all878_original_TAB_case_scope_and_captured_status.csv','five_shared_original_SMD_comparison_references.csv']:
    assert sha(ROOT/s.BASE/name)==sha(out/name)
xml=ROOT/'tmp/catalog_v65_validation_20261016_v1.xml';suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==8
assert all(int(x.get(k,0))==0 for x in suites for k in ['failures','errors','skipped'])
shutil.copyfile(xml,out/'relevant_fresh8_validation.xml')
shutil.copyfile(ROOT/s.CONFIG,out/'frozen_publication_configuration.json')
shutil.copyfile(__file__,out/'packaging_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
page=ROOT/'projects/flow_matching_research/README.rst';text=page.read_text(encoding='utf-8')
for a,b in [
    ('2026-10-16 的 v64 增量快照','2026-10-16 的 v65 增量快照'),
    ('all_algorithm_dataset_metrics_v64/','all_algorithm_dataset_metrics_v65/'),
    ('共11925行指标，3171行有数值','共11925行指标，3186行有数值'),
    ('原v63的11910行全部字段和捕获时间保持；仅填入CrossAD第20条件的15项原待完成指标。','原v64的11910行全部字段和捕获时间保持；仅填入CrossAD第21条件的15项原待完成指标。'),
    ('该快照已核验分别20和8个条件','该快照已核验分别21和8个条件')]:
    assert a in text,a;text=text.replace(a,b)
text+='''
`完整算法—数据集指标表更新至v65 <results/2026-10-16/all_algorithm_dataset_metrics_v65/README.md>`_
11925行全部保留，3186行有数值；仅填入CrossAD row4/seed1105的15项原待完成控制，其他11910行及捕获时间保持。
CrossAD21/139、完整主要组1/22；70生成任务、878原TAB案例及全部范围保持，8项增量保护检查通过。
TEP95、MOMENT22和GRASP51等独立证据包保留各自轨道与时间戳，本表不冒充全部进程实时状态，整体未完成。
'''
page.write_text(text,encoding='utf-8',newline='\n')
ara=ROOT/'projects/flow_matching_research/ara/execution/CrossAD_row4_catalog.2026-10-16_v65.json'
assert not ara.exists();write(ara,dict(packet=out.relative_to(ROOT).as_posix(),metric_rows=11925,numeric_rows=3186,
    unchanged_rows_all_fields=11910,new_complete_fixed_case_controls=15,all_original_experiments_complete=False))
manifest=read(ROOT/'tmp/CrossAD_row4_seed1105_full21_publication_manifest_20261016_v1.json')
head=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
assert head(ROOT)==manifest['previous_root'] and head(ROOT.parent/'flow-matching-research')==manifest['previous_parent']
paths=set(manifest['paths'])|{s.MODULE.replace('.','/')+'.py',s.TEST,s.CONFIG,page.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix()}
paths|={p.relative_to(ROOT).as_posix() for p in out.rglob('*') if p.is_file()}
write(ROOT/'tmp/CrossAD21_and_catalog_v65_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(publication_files=len(paths),CrossAD_full_conditions=21,metric_rows=11925,numeric_rows=3186,changed_rows=15)))
