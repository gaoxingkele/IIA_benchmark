"""Publish two complete model closures and only the newCrossAD fixed cells."""
import csv
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import publish_crossad_row4_native_increment_v71 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

c=read(ROOT/s.CONFIG);assert c==s.registration()
base=ROOT/s.BASE;out=ROOT/c['output_directory']
def csvrows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
before=csvrows(base/'all_algorithm_dataset_metrics.csv');after=csvrows(out/'all_algorithm_dataset_metrics.csv')
assert len(before)==len(after)==11925 and sum(bool(r['mean']) for r in after)==3269
assert sum(a!=b for a,b in zip(before,after))==15
assert all(a==b for a,b in zip(before,after) if s.IDENTITY not in a['protocol'])
assert all(not a['mean'] and b['mean'] for a,b in zip(before,after) if a!=b)
for name in ['all878_original_TAB_case_scope_and_captured_status.csv','five_shared_original_SMD_comparison_references.csv']:
    assert (base/name).read_bytes()==(out/name).read_bytes()
for packet in [ROOT/s.PACKET,ROOT/'projects/flow_matching_research/results/2026-10-16/GRASP_Lin_SwanSF_seed1107_full53_v1']:
    for r in read(packet/'packet_integrity.json')['files']:assert sha(packet/r['path'])==r['sha256']
xml=ROOT/'tmp/CrossAD_row4_catalog_v71_validation_20261016_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'));assert sum(int(v.get('tests',0)) for v in suites)==8
assert all(int(v.get(k,0))==0 for v in suites for k in ['failures','errors','skipped'])
shutil.copyfile(xml,out/'fresh8_increment_preservation_validation.xml')
assert '<title>实验指标 v71</title>' in (out/'results.html').read_text(encoding='utf-8')
shutil.copyfile(__file__,out/'packaging_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/all_algorithm_dataset_metrics_v71.2026-10-16.json'
write(ara,dict(catalog=c['output_directory'],previous_catalog=s.BASE,completed_new_case=s.IDENTITY,
    metric_rows=11925,numeric_rows=3269,previously_pending_controls_completed=15,other11910_rows_and_capture_times_preserved=True,
    CrossAD_complete_conditions=26,CrossAD_required_conditions=139,generation_complete_reexecutions=10,generation_required_reexecutions=70,
    generation_complete_five_model_cohorts=2,generation_required_cohorts=14,
    separate_GRASP_packet_complete_models=53,required_GRASP_models=9120,GRASP_partial_ten_seed_statistics_computed=False,
    original_scope=c['original_scope'],new_numerical_executions_by_publication=0,all_original_experiments_complete=False))
page=ROOT/'projects/flow_matching_research/README.rst';text=page.read_text(encoding='utf-8')
prefix,rest=text.split('原论文单细胞轨道另见',1)
assert 'all_algorithm_dataset_metrics_v70' in prefix
prefix=prefix.replace('all_algorithm_dataset_metrics_v70','all_algorithm_dataset_metrics_v71').replace('的 v70 增量快照','的 v71 增量快照')
prefix=prefix.replace('3254行有数值','3269行有数值')
prefix=prefix.replace('原v69的11906行全部字段和捕获时间保持；仅填入CrossAD第25条件的15项和生成重执行第10条件的4项原待完成指标。',
    '原v70的11910行全部字段和捕获时间保持；仅填入CrossAD第26条件的15项原待完成指标。')
prefix=prefix.replace('分别25和10个条件','分别26和10个条件')
page.write_text(prefix+'原论文单细胞轨道另见'+rest,encoding='utf-8',newline='\n')
manifests=[read(ROOT/p) for p in ['tmp/GRASP_Lin1107_full53_publication_manifest_20261016_v1.json','tmp/CrossAD_row4_seed1106_full26_manifest_20261016_v1.json']]
assert all(m['previous_root']==manifests[0]['previous_root'] and m['previous_parent']==manifests[0]['previous_parent'] for m in manifests)
paths=set().union(*(set(m['paths']) for m in manifests))
paths.update([s.CONFIG,s.MODULE.replace('.','/')+'.py',s.TEST,ara.relative_to(ROOT).as_posix()])
paths.update(p.relative_to(ROOT).as_posix() for p in out.iterdir() if p.is_file())
head=lambda d:subprocess.check_output(['git','-C',str(d),'rev-parse','HEAD'],text=True).strip()
assert head(ROOT)==manifests[0]['previous_root'] and head(ROOT.parent/'flow-matching-research')==manifests[0]['previous_parent']
write(ROOT/'tmp/GRASP53_CrossAD26_catalog_v71_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(dict(files=len(paths),metric_rows=11925,numeric_rows=3269,CrossAD_complete_conditions=26,GRASP_complete_models=53))
