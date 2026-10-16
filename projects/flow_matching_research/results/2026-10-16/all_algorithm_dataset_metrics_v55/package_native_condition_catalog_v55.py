from pathlib import Path
import csv
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import publish_native_condition_catalog_v55 as s
ROOT=s.ROOT
c=s.read(ROOT/s.CONFIG);base=ROOT/c['state_root'];output=ROOT/c['output_directory']
guard=s.read(base/'guard_receipt.json');report=s.read(output/'publication_validation.json')
assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted']
assert not guard['model_training_or_inference_executed'] and guard['report_sha256']==s.sha(output/'publication_validation.json')
assert report['metric_rows']==10991 and report['numeric_rows']==2320 and report['new_numeric_cells']==189
assert report['configuration_sha256']==s.sha(ROOT/s.CONFIG)
for item in c['source_receipts']:assert s.sha(ROOT/item['path'])==item['sha256']
def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
before,after=rows(ROOT/c['base_csv']),rows(output/'all_algorithm_dataset_metrics.csv')
s.validate(before,after,11,6)
assert before==after[:8626]
for item in after[8626:]:
    if item['mean']:
        proof=s.read(item['source']);recipe=json.loads(item['score_recipe']);cid=recipe['canonical_id']
        if item['task']=='anomaly_detection':
            ratio=item['protocol'].split('/validation_',1)[1].split('percent',1)[0]
            field={'F1':'f1','AUROC':'auroc','AP':'average_precision'}.get(item['metric'],item['metric'])
            assert float(item['mean'])==proof['independent_strict_metrics'][ratio]['micro'][field]
        else:
            metric=proof['metrics'][item['metric']]
            assert float(item['mean'])==metric['mean']
            assert float(item['evaluator_std'])==metric['std']
            assert json.loads(item['evaluator_ci95'])==[metric['ci95_low'],metric['ci95_high']]
tests=ROOT/'tmp/native_condition_catalog_v55_tests.xml'
suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==12
assert not any(int(x.get(k,0)) for x in suites for k in ['failures','errors','skipped'])
launch=s.read(base/'launch.json');assert launch['validation_sha256']==s.sha(tests)
(output/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8')
for name,path in dict(actual_guard=base/'guard_receipt.json',actual_controller_status=base/'status.json',
                      actual_launch=base/'launch.json').items():
    shutil.copyfile(path,output/(name+'.json'))
shutil.copyfile(tests,output/'validation_12_cases.xml')
for name in ['run_native_condition_catalog_v55_guarded.py','package_native_condition_catalog_v55.py']:
    shutil.copyfile(ROOT/'tmp'/name,output/name)
(output/'freshness_scope.json').write_text(json.dumps(dict(
    old8626_rows_exactly_preserved=True,other_algorithm_rows_retain_original_capture_times=True,
    newly_merged_tracks=['CrossAD_original_CUDA_FP32_full139_v2','full70_generation_evaluator_reexecution_v1'],
    frozen_new_track_CrossAD_conditions=11,frozen_new_track_generation_conditions=6,
    partial_or_future_outputs_not_imported=True,all_recent_independent_packets_merged=False,
    metric_counts_are_not_training_or_experiment_completion_counts=True,
    all_original_experiments_complete=False),indent=2)+'\n',encoding='utf-8')
proof=dict(files=[dict(path=p.relative_to(output).as_posix(),sha256=s.sha(p))
    for p in sorted(output.rglob('*')) if p.is_file()])
(output/'packet_integrity.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
ara=ROOT/'projects/flow_matching_research/ara/execution/native_condition_catalog.2026-10-16_v55.json'
assert not ara.exists()
ara.write_text(json.dumps(dict(result_packet=output.relative_to(ROOT).as_posix(),metric_rows=10991,
    numeric_rows=2320,legacy_rows_preserved=8626,tests_passed=12,
    new_complete_benchmark_conditions=0,all_original_experiments_complete=False),indent=2)+'\n',encoding='utf-8')
paths=[s.CONFIG,s.MODULE.replace('.','/')+'.py',s.TEST,ara.relative_to(ROOT).as_posix(),
       'projects/flow_matching_research/README.rst']+[p.relative_to(ROOT).as_posix() for p in output.rglob('*') if p.is_file()]
head=lambda path:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD']).decode().strip()
manifest=dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths))
target=ROOT/'tmp/native_condition_catalog_v55_manifest.json';assert not target.exists()
target.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(files=len(paths),rows=len(after),numeric_rows=2320,old_rows_unchanged=True,tests_passed=12)))
