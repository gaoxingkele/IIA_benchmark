import csv
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import publish_SPOT_full25_condition_catalog_v61 as s

ROOT=s.ROOT;c=s.read(ROOT/s.CONFIG);assert c==s.registration()
out=ROOT/c['output_directory'];before=ROOT/c['base_catalog']
with (out/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
    reader=csv.DictReader(f);rows=list(reader);fields=reader.fieldnames
with (before/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
    reader=csv.DictReader(f);old=list(reader);old_fields=reader.fieldnames
assert len(rows)==11925 and len(old)==11795 and sum(bool(r['mean']) for r in rows)==3107
changed=[]
for index,(a,b) in enumerate(zip(old,rows)):
    if a!={k:b[k] for k in old_fields}:
        assert a['n']=='0' and a['mean']=='' and b['n']=='1' and b['mean']
        assert s.CROSS_ID in b['protocol'] and s.p.key(a)==s.p.key(b)
        changed.append(index)
assert len(changed)==15
added=rows[11795:];assert len(added)==130
assert sum(r['n']=='1' and not r['mean'] for r in added)==16
assert sum(r['n']=='0' for r in added)==106
assert all(not r[k] for r in added for k in ['std','se','ci95_low','ci95_high','entity_bootstrap_ci95'])
assert s.sha(out/'all878_original_TAB_case_scope_and_captured_status.csv')==s.sha(before/'all878_original_TAB_case_scope_and_captured_status.csv')
with (out/'five_shared_original_SMD_comparison_references.csv').open(encoding='utf-8-sig',newline='') as f:refs=list(csv.DictReader(f))
assert len(refs)==5 and all(r['new_independent_case']=='False' for r in refs)
xml=Path('F:/aicoding/IIA_Data/experiments/SPOT_full25_condition_catalog_v61_20261016_v1_tests.xml')
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==8
assert all(int(x.get('failures',0))==int(x.get('errors',0))==int(x.get('skipped',0))==0 for x in suites)
shutil.copyfile(xml,out/'relevant8_scope_and_undefined_macro_checks.xml')
shutil.copyfile(__file__,out/'packaging_source.py')
(out/'packet_integrity.json').write_text(json.dumps(dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=s.sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]),indent=2)+'\n',encoding='utf-8')
ara=ROOT/'projects/flow_matching_research/ara/execution/SPOT_full25_condition_catalog.2026-10-16_v61.json';assert not ara.exists()
ara.write_text(json.dumps(dict(packet=out.relative_to(ROOT).as_posix(),metric_rows=11925,numeric_rows=3107,
    CrossAD_complete_conditions=16,SPOT_complete_cases=3,SPOT_required_cases=25,SPOT_required_comparison_slots=30,
    SPOT_actual_controls=130,SPOT_complete_defined_controls=8,SPOT_complete_undefined_controls=16,SPOT_pending_controls=106,
    all_original_experiments_complete=False),indent=2)+'\n',encoding='utf-8')
paths=[s.CONFIG,s.TEST,s.MODULE.replace('.','/')+'.py',ara.relative_to(ROOT).as_posix(),'projects/flow_matching_research/README.rst']
paths+=[p.relative_to(ROOT).as_posix() for p in out.rglob('*') if p.is_file()]
head=lambda path:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD']).decode().strip()
manifest=ROOT/'tmp/SPOT_catalog_v61_manifest_20261016_v1.json';assert not manifest.exists()
manifest.write_text(json.dumps(dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(metric_rows=11925,numeric_rows=3107,changed_legacy_pending_controls=15,added_full_scope_controls=130,files=len(paths))))
