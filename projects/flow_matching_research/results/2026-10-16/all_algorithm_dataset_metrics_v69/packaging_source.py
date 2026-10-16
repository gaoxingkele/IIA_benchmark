"""Publish the next actual complete CUDA case and its fifteen existing slots."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import publish_crossad_row2_native_increment_v69 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

c=read(ROOT/s.CONFIG);assert c==s.registration()
base=ROOT/s.BASE;out=ROOT/c['output_directory']
with (base/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:before=list(csv.DictReader(f))
with (out/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:after=list(csv.DictReader(f))
assert len(before)==len(after)==11925
assert sum(bool(r['mean']) for r in before)==3220 and sum(bool(r['mean']) for r in after)==3235
changes=[(a,b) for a,b in zip(before,after) if a!=b];assert len(changes)==15
assert all(not a['mean'] and b['mean'] and json.loads(b['score_recipe'])['canonical_id']==s.IDENTITY for a,b in changes)
assert all(a==b for a,b in zip(before,after) if s.IDENTITY not in a['protocol'])
for name in ['all878_original_TAB_case_scope_and_captured_status.csv','five_shared_original_SMD_comparison_references.csv']:
    assert (out/name).read_bytes()==(base/name).read_bytes()
xml=ROOT/'tmp/CrossAD_row2_catalog_increment_v69_validation_20261016_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(v.get('tests',0)) for v in suites)==8 and all(int(v.get(k,0))==0 for v in suites for k in ['failures','errors','skipped'])
shutil.copyfile(xml,out/'fresh8_increment_preservation_validation.xml')
html=(out/'results.html').read_text(encoding='utf-8')
assert '<title>实验指标 v69</title>' in html and '<h1>算法—数据集—指标 v69</h1>' in html
assert all(r['protocol'] in html for a,r in changes)
packet=ROOT/s.PACKET
for r in read(packet/'packet_integrity.json')['files']:assert sha(packet/r['path'])==r['sha256']
assert read(packet/'iteration_status.json')['CrossAD_full_CUDA_conditions']==24
shutil.copyfile(__file__,out/'packaging_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/all_algorithm_dataset_metrics_v69.2026-10-16.json'
write(ara,dict(catalog=c['output_directory'],completed_case=s.IDENTITY,previous_catalog=s.BASE,
    all_metric_rows=11925,numeric_rows=3235,actual_newly_completed_controls=15,other11910_rows_preserved=True,
    full_CrossAD_CUDA_conditions=24,required_CrossAD_conditions=139,original_scope=c['original_scope'],
    publication_started_no_numerical_workers=True,all_original_experiments_complete=False))
old=read(ROOT/'tmp/CrossAD_row2_seed1106_full24_publication_manifest_20261016_v1.json')
paths=set(old['paths'])|{p.relative_to(ROOT).as_posix() for p in out.iterdir() if p.is_file()}
paths.update([s.CONFIG,s.MODULE.replace('.','/')+'.py',s.TEST,ara.relative_to(ROOT).as_posix()])
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==old['previous_root']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent/'flow-matching-research',text=True).strip()==old['previous_parent']
write(ROOT/'tmp/CrossAD24_and_catalog_v69_publication_manifest_20261016_v1.json',dict(previous_root=old['previous_root'],previous_parent=old['previous_parent'],paths=sorted(paths)))
print(json.dumps(dict(files=len(paths),metric_rows=11925,numeric_rows=3235,CrossAD_conditions=24,prior_rows_unchanged=11910)))
