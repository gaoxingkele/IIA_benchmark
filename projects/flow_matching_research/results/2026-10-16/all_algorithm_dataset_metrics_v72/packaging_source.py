"""Freeze the exact CrossAD27 delta and preserve all unrelated catalog rows."""
import csv
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import publish_crossad_row5_native_increment_v72 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

c=read(ROOT/s.CONFIG);assert c==s.registration()
base=ROOT/s.BASE;out=ROOT/c['output_directory']
def csvrows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
before=csvrows(base/'all_algorithm_dataset_metrics.csv');after=csvrows(out/'all_algorithm_dataset_metrics.csv')
assert len(before)==len(after)==11925 and sum(bool(r['mean']) for r in before)==3269 and sum(bool(r['mean']) for r in after)==3284
assert sum(a!=b for a,b in zip(before,after))==15
assert all(a==b for a,b in zip(before,after) if s.IDENTITY not in a['protocol'])
assert all(not a['mean'] and b['mean'] for a,b in zip(before,after) if a!=b)
for name in ['all878_original_TAB_case_scope_and_captured_status.csv','five_shared_original_SMD_comparison_references.csv']:
    assert (base/name).read_bytes()==(out/name).read_bytes()
packet=ROOT/s.PACKET
for item in read(packet/'packet_integrity.json')['files']:assert sha(packet/item['path'])==item['sha256']
xml=ROOT/'tmp/CrossAD_row5_catalog_v72_validation_20261016_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(v.get('tests',0)) for v in suites)==9 and not any(int(v.get(k,0)) for v in suites for k in ['failures','errors','skipped'])
assert '<title>实验指标 v72</title>' in (out/'results.html').read_text(encoding='utf-8')
shutil.copyfile(xml,out/'fresh9_increment_and_current_test_provenance_validation.xml')
shutil.copyfile(__file__,out/'packaging_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/all_algorithm_dataset_metrics_v72.2026-10-16.json'
assert not ara.exists()
write(ara,dict(catalog=c['output_directory'],previous_catalog=s.BASE,completed_new_case=s.IDENTITY,
    metric_rows=11925,numeric_rows=3284,previously_pending_controls_completed=15,other11910_rows_and_capture_times_preserved=True,
    CrossAD_complete_conditions=27,CrossAD_required_conditions=139,generation_complete_reexecutions=10,generation_required_reexecutions=70,
    generation_complete_five_model_cohorts=2,generation_required_cohorts=14,
    original_scope=c['original_scope'],new_numerical_executions_by_publication=0,all_original_experiments_complete=False))
page=ROOT/'projects/flow_matching_research/README.rst'
text=page.read_text(encoding='utf-8');prefix,rest=text.split('原论文单细胞轨道另见',1)
assert 'all_algorithm_dataset_metrics_v71' in prefix
prefix=prefix.replace('all_algorithm_dataset_metrics_v71','all_algorithm_dataset_metrics_v72').replace('的 v71 增量快照','的 v72 增量快照')
prefix=prefix.replace('3269行有数值','3284行有数值').replace('原v70的11910行','原v71的11910行').replace('第26条件','第27条件').replace('分别26和10个条件','分别27和10个条件')
page.write_text(prefix+'原论文单细胞轨道另见'+rest,encoding='utf-8',newline='\n')
manifest=read(ROOT/'tmp/CrossAD_row5_seed1106_full27_manifest_20261016_v1.json')
head=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
assert head(ROOT)==manifest['previous_root'] and head(ROOT.parent/'flow-matching-research')==manifest['previous_parent']
paths=set(manifest['paths'])|{s.CONFIG,s.MODULE.replace('.','/')+'.py',s.TEST,ara.relative_to(ROOT).as_posix()}
paths.update(p.relative_to(ROOT).as_posix() for p in out.iterdir() if p.is_file())
write(ROOT/'tmp/CrossAD27_catalog_v72_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(dict(files=len(paths),metric_rows=11925,numeric_rows=3284,CrossAD_complete_conditions=27))
