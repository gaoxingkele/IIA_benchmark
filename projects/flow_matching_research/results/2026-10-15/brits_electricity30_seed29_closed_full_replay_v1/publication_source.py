"""Publish the completed seed29 model with full native/independent receipts."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from scripts.flow_matching.prepare_tsad_execution import sha

root=Path(__file__).resolve().parents[1]
base='projects/flow_matching_research/results/2026-10-15/brits_electricity30_seed29_closed_full_replay_v1'
folder=root/base
closure=json.loads((folder/'actual_original_brits_electricity30_seed29_complete_closure.json').read_text())
assert closure['full_original_training_and_model_reinference_and_independent_array_review_passed']
assert closure['next_epoch']==995 and closure['best_epoch_source_index']==963
assert closure['verified_SAITS_BRITS_models']==38 and closure['completed_verified_seeds_of_this_rate']==[26,27,28,29]
assert closure['metric_tolerance']==dict(rtol=2e-6,atol=2e-6)
assert not closure['five_seed_aggregate_complete'] and not closure['all_original_experiments_complete']
xml=root/'tmp/BRITS_Electricity30_seed29_full38_validation_20261015_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==40
assert all(int(s.get('failures',0))==int(s.get('errors',0))==0 for s in suites)
shutil.copyfile(xml,folder/'validation.xml')
(folder/'publication_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
with (folder/'README.md').open('a',encoding='utf-8',newline='\n') as stream:
    stream.write('\n40 relevant validation checks passed. Complete reviewed SAITS/BRITS models:38/210; '
        'BRITS nominal30% Electricity seeds26,27,28,29:4/5. Seed30 remains required.\n')
job=closure['id'];proofbase='projects/flow_matching_research/results/2026-10-11/saits_automatic_complete_model_reviews_v1/'+job
targets=[base,proofbase,
    'configs/reproducibility/fm_saits_completion_replay_queue_v1/'+job+'.native.json',
    'configs/reproducibility/fm_saits_completion_replay_queue_v1/'+job+'.review.json']
paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    raw=subprocess.check_output(['git',*args,'--',*targets],cwd=root)
    paths.update(p for p in raw.decode().splitlines() if not p.endswith('.log'))
assert paths and not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root).strip()
manifest=dict(previous_root='3bea904083883a2fa747b420ee5959c4633e87fd',
    previous_parent='e57076e8656df7a4cd31b32b3a9a88694c1486cd',paths=sorted(paths))
(root/'tmp/BRITS30_seed29_full38_publication_manifest_20261015_v1.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(exact_files=len(paths),complete_reviewed_original_models=38,required_original_models=210,
    nominal30_closed_original_seeds=[26,27,28,29],validation_tests_passed=40)))
