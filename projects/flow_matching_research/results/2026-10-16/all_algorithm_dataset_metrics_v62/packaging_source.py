from pathlib import Path
import csv
import json
import shutil
import xml.etree.ElementTree as ET
from scripts.flow_matching import publish_verified_native_increment_catalog_v62 as s
from scripts.flow_matching.timegan_native_common_v1 import write

ROOT=s.ROOT;c=s.read(ROOT/s.CONFIG);assert c==s.registration();out=ROOT/c['output_directory']
with (ROOT/c['base_catalog']/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:old=list(csv.DictReader(f))
with (out/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:new=list(csv.DictReader(f))
assert len(old)==len(new)==11925 and sum(bool(r['mean']) for r in new)==3126
changed=[(a,b) for a,b in zip(old,new) if a!=b];assert len(changed)==19
assert all(a['n']=='0' and not a['mean'] and b['n']=='1' and b['mean'] and s.p.key(a)==s.p.key(b) for a,b in changed)
assert sum(s.CROSS_ID in b['protocol'] for _,b in changed)==15
assert sum(s.GEN_ID in b['protocol'] for _,b in changed)==4
for name in ['all878_original_TAB_case_scope_and_captured_status.csv','five_shared_original_SMD_comparison_references.csv']:
    assert s.sha(out/name)==s.sha(ROOT/c['base_catalog']/name)
xml=Path('F:/aicoding/IIA_Data/experiments/verified_native_increment_catalog_v62_20261016_v1_tests.xml')
suites=list(ET.parse(xml).getroot().iter('testsuite'));assert sum(int(x.get('tests',0)) for x in suites)==8
assert all(int(x.get('errors',0))==int(x.get('failures',0))==int(x.get('skipped',0))==0 for x in suites)
shutil.copyfile(xml,out/'relevant8_pending_control_protection_checks.xml');shutil.copyfile(__file__,out/'packaging_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=s.sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/verified_native_increment_catalog.2026-10-16_v62.json'
assert not ara.exists();write(ara,dict(packet=out.relative_to(ROOT).as_posix(),metric_rows=11925,numeric_rows=3126,
    prior11906_rows_all_fields_and_capture_times_unchanged=True,newly_completed_pending_controls=19,
    CrossAD_complete_conditions=17,generation_complete_reexecution_conditions=8,all_original_experiments_complete=False))
manifest=ROOT/'tmp/Spectral_Energy44_full8_manifest_20261016_v1.json';m=s.read(manifest)
paths=[s.CONFIG,s.TEST,s.MODULE.replace('.','/')+'.py',ara.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in out.rglob('*') if p.is_file()]
m['paths']=sorted(set(m['paths'])|set(paths));combined=ROOT/'tmp/Spectral8_and_catalog_v62_manifest_20261016_v1.json'
assert not combined.exists();write(combined,m)
print(json.dumps(dict(files=len(m['paths']),generation_complete_conditions=8,CrossAD_complete_conditions=17,
    numeric_rows=3126,unchanged_rows=11906,manifest=str(combined))))
