from pathlib import Path
import csv
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import publish_crossad_two_native_increments_v63 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

out=ROOT/read(ROOT/s.CONFIG)['output_directory']
with (ROOT/s.BASE/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:before=list(csv.DictReader(f))
with (out/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:after=list(csv.DictReader(f))
assert len(before)==len(after)==11925 and sum(bool(x['mean']) for x in after)==3156
changed=[b for a,b in zip(before,after) if a!=b]
assert len(changed)==30 and s.merge(before,changed)==after
for name in ['all878_original_TAB_case_scope_and_captured_status.csv','five_shared_original_SMD_comparison_references.csv']:
    assert sha(ROOT/s.BASE/name)==sha(out/name)
xml=ROOT/'tmp/catalog_v63_validation_20261016_v1.xml';suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==17
assert all(int(x.get('failures',0))==int(x.get('errors',0))==int(x.get('skipped',0))==0 for x in suites)
shutil.copyfile(xml,out/'relevant_fresh17_validation.xml')
shutil.copyfile(ROOT/s.CONFIG,out/'frozen_publication_configuration.json')
shutil.copyfile(__file__,out/'packaging_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
page=ROOT/'projects/flow_matching_research/README.rst'
page.write_text(page.read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
ara=ROOT/'projects/flow_matching_research/ara/execution/CrossAD_two_native_increments_catalog.2026-10-16_v63.json'
assert not ara.exists();write(ara,dict(packet=out.relative_to(ROOT).as_posix(),all_metric_rows=11925,numeric_rows=3156,
    original_rows_and_capture_times_preserved=11895,previously_pending_controls_filled=30,
    full_CrossAD_conditions=19,required_CrossAD_conditions=139,all_original_experiments_complete=False))
paths={s.MODULE.replace('.','/')+'.py',s.TEST,s.CONFIG,page.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix()}
paths|={p.relative_to(ROOT).as_posix() for p in out.rglob('*') if p.is_file()}
for manifest in ['tmp/CrossAD_row2_seed1105_full19_publication_manifest_20261016_v1.json','tmp/CrossAD_common3seed_protocol_diagnostics_manifest_20261016_v1.json']:
    m=read(ROOT/manifest);paths.update(m['paths'])
    assert m['previous_root']=='f4bab1c27d315dd7a2ad0fe437215b52d6c88c79'
    assert m['previous_parent']=='dd26346a8cc446e8f1cee8a9adc3a640882a2036'
head=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
write(ROOT/'tmp/CrossAD19_diagnostics_catalog_v63_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(files=len(paths),all_rows=11925,numeric_rows=3156,actual_CrossAD_conditions=19)))
