"""Publish two actually completed full conditions and exactly nineteen cells."""
import csv
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import publish_CrossAD25_Spectral10_native_increment_v70 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

c=read(ROOT/s.CONFIG);assert c==s.registration()
base=ROOT/s.BASE;out=ROOT/c['output_directory']
def load(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
before=load(base/'all_algorithm_dataset_metrics.csv');after=load(out/'all_algorithm_dataset_metrics.csv')
assert len(before)==len(after)==11925 and sum(bool(r['mean']) for r in after)==3254
changed=[(a,b) for a,b in zip(before,after) if a!=b];assert len(changed)==19
assert all(not a['mean'] and b['mean'] for a,b in changed)
assert all(a==b for a,b in zip(before,after) if s.CROSS_ID not in a['protocol'] and s.GEN_ID not in a['protocol'])
for name in ['all878_original_TAB_case_scope_and_captured_status.csv','five_shared_original_SMD_comparison_references.csv']:
    assert (base/name).read_bytes()==(out/name).read_bytes()
for packet_name in [s.CROSS_PACKET,s.GEN_PACKET]:
    packet=ROOT/packet_name
    for r in read(packet/'packet_integrity.json')['files']:assert sha(packet/r['path'])==r['sha256']
    assert not read(packet/'iteration_status.json')['all_original_experiments_complete']
xml=ROOT/'tmp/CrossAD25_Spectral10_catalog_v70_validation_20261016_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(v.get('tests',0)) for v in suites)==11 and all(int(v.get(k,0))==0 for v in suites for k in ['failures','errors','skipped'])
shutil.copyfile(xml,out/'fresh11_increment_preservation_validation.xml')
html=(out/'results.html').read_text(encoding='utf-8')
assert '<title>实验指标 v70</title>' in html and all(b['protocol'] in html for a,b in changed)
shutil.copyfile(__file__,out/'packaging_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/all_algorithm_dataset_metrics_v70.2026-10-16.json'
write(ara,dict(catalog=c['output_directory'],completed_new_conditions=[s.CROSS_ID,s.GEN_ID],previous_catalog=s.BASE,
    original_scope=c['original_scope'],all_metric_rows=11925,numeric_rows=3254,newly_completed_metric_controls=19,
    other11906_rows_and_capture_times_preserved=True,complete_CrossAD_conditions=25,required_CrossAD_conditions=139,
    complete_generation_reexecutions=10,required_generation_reexecutions=70,complete_five_model_generation_cohorts=2,
    required_five_model_generation_cohorts=14,new_numerical_workers_by_publication=0,all_original_experiments_complete=False))
manifests=[read(ROOT/p) for p in ['tmp/CrossAD_row3_seed1106_full25_manifest_20261016_v1.json','tmp/Spectral_Energy46_full10_manifest_20261016_v1.json']]
assert all(m['previous_root']==manifests[0]['previous_root'] and m['previous_parent']==manifests[0]['previous_parent'] for m in manifests)
paths=set().union(*(set(m['paths']) for m in manifests))
paths.update([s.CONFIG,s.MODULE.replace('.','/')+'.py',s.TEST,ara.relative_to(ROOT).as_posix()])
paths.update(p.relative_to(ROOT).as_posix() for p in out.iterdir() if p.is_file())
head=lambda d:subprocess.check_output(['git','-C',str(d),'rev-parse','HEAD'],text=True).strip()
assert head(ROOT)==manifests[0]['previous_root'] and head(ROOT.parent/'flow-matching-research')==manifests[0]['previous_parent']
write(ROOT/'tmp/CrossAD25_Spectral10_catalog_v70_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(dict(files=len(paths),metric_rows=11925,numeric_rows=3254,new_controls=19,CrossAD_complete=25,generation_complete=10,generation_complete_five_model_groups=2))
