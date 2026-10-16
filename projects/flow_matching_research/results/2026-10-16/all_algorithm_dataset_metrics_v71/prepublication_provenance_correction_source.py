"""Correct the inherited test pin before either repository publishes v71.

The exact staged-file gate caught a tracked old test instead of the new test.
Preserve the prior publisher/config and validation; retain every metric byte.
"""
import csv
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from scripts.flow_matching import publish_crossad_row4_native_increment_v71 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

oldconfig=read(ROOT/s.CONFIG);out=ROOT/oldconfig['output_directory']
assert oldconfig['source_receipts'][1]['path']=='tests/test_fm_crossad_row2_native_increment_v69.py'
for r in read(out/'packet_integrity.json')['files']:assert sha(out/r['path'])==r['sha256']
fixed_data=[out/'all_algorithm_dataset_metrics.csv',out/'results.html',out/'publication_validation.json']
before={p.name:sha(p) for p in fixed_data}
shutil.copyfile(ROOT/s.CONFIG,out/'preserved_prepublication_inherited_test_pin_configuration.json')
shutil.copyfile(out/'publication_source.py',out/'preserved_prepublication_inherited_test_pin_source.py')
newconfig=s.registration();assert newconfig['source_receipts'][1]['path']==s.TEST
write(ROOT/s.CONFIG,newconfig);shutil.copyfile(ROOT/(s.MODULE.replace('.','/')+'.py'),out/'publication_source.py')
xml=ROOT/'tmp/CrossAD_row4_catalog_v71_validation_20261016_v2.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==9 and all(int(x.get(k,0))==0 for x in suites for k in ['failures','errors','skipped'])
shutil.copyfile(xml,out/'fresh9_current_test_provenance_validation.xml')
with (ROOT/s.BASE/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
    reader=csv.DictReader(f);base=list(reader);headers=reader.fieldnames
with (out/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:published=list(csv.DictReader(f))
captured=read(out/'publication_validation.json')['captured_utc']
assert s.merge(base,s.actual_updates(headers,captured))==published
assert before=={p.name:sha(p) for p in fixed_data}
proof=dict(corrected_before_Git_publication=True,previous_test_pin=oldconfig['source_receipts'][1]['path'],current_test_pin=s.TEST,
    current_test_file_sha256=sha(ROOT/s.TEST),current_frozen_configuration_sha256=sha(ROOT/s.CONFIG),
    fresh_actual_current_module_tests=9,all_metric_CSV_HTML_and_capture_bytes_unchanged=before,
    actual_full_condition_evidence_rechecked=True,new_training_or_inference_executions=0)
write(out/'prepublication_test_provenance_correction.json',proof)
shutil.copyfile(__file__,out/'prepublication_provenance_correction_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file() and p.name!='packet_integrity.json']))
ara=ROOT/'projects/flow_matching_research/ara/execution/all_algorithm_dataset_metrics_v71.2026-10-16.json';a=read(ara)
a.update(current_test_receipt=s.TEST,current_frozen_publication_configuration_sha256=sha(ROOT/s.CONFIG),
    fresh_current_test_provenance_tests=9,prepublication_inherited_test_pin_corrected=True)
write(ara,a)
manifest_path=ROOT/'tmp/GRASP53_CrossAD26_catalog_v71_publication_manifest_20261016_v1.json';m=read(manifest_path)
paths=set(m['paths']);paths.remove('tests/test_fm_crossad_row2_native_increment_v69.py');paths.add(s.TEST)
paths.update(p.relative_to(ROOT).as_posix() for p in out.iterdir() if p.is_file());m['paths']=sorted(paths);write(manifest_path,m)
assert read(ROOT/s.CONFIG)==s.registration()
for r in read(out/'packet_integrity.json')['files']:assert sha(out/r['path'])==r['sha256']
print(dict(exact_files=len(paths),fresh_current_tests=9,metric_rows=11925,numeric_rows=3269,all_metric_and_capture_bytes_preserved=True))
