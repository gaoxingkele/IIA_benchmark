"""Publish one complete mean variant with the full39-model ledger."""
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

root=Path(__file__).resolve().parents[1]
base='projects/flow_matching_research/results/2026-10-15/grasp_mean_seed1106_full39_model_increment_v1'
folder=root/base
closure=json.loads((folder/'actual_full39_model_increment_closure.json').read_text())
assert closure['complete_original_training_and_full_native_replay_and_independent_recipe_review']
assert closure['actual_current_unique_complete_models']==39 and closure['parameter_count']==69244
assert closure['epochs']==1500 and closure['optimizer_updates']==93000 and len(closure['recipes'])==1
assert closure['complete_mean_Swan_seeds']==[1103,1104,1105,1106]
assert closure['required_original_entity_models']==9120 and not closure['full_ten_seed_mean_cohort_complete']
assert not closure['all_original_experiments_complete'] and not closure['strict_TAB_result']
xml=root/'tmp/GRASP_mean1106_full39_validation_20261014_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==34
assert all(int(s.get('failures',0))==int(s.get('errors',0))==0 for s in suites)
shutil.copyfile(xml,folder/'validation.xml')
(folder/'publication_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
job=closure['canonical_id']
configroot='configs/reproducibility/fm_grasp_completion_replay_children_v1'
configs=[p.relative_to(root).as_posix() for p in (root/configroot).glob(job+'*.json')]
assert len(configs)>=3
targets=[base,'projects/flow_matching_research/results/2026-10-12/grasp_full9120_completion_replay_v1/completed_models/'+job,*configs]
paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    raw=subprocess.check_output(['git',*args,'--',*targets],cwd=root)
    paths.update(p for p in raw.decode().splitlines() if not p.endswith('.log'))
assert paths and not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root).strip()
manifest=dict(previous_root='73cf0d1da26c039e3aeb161440da8f974445ee41',
    previous_parent='41959eef904716e091384f31094d40da73853cd1',paths=sorted(paths))
(root/'tmp/GRASP_mean1106_full39_publication_manifest_20261015_v1.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(exact_files=len(paths),complete_original_models=39,required_original_models=9120,
    original_mean_SWAN_seeds_closed=[1103,1104,1105,1106],validation_tests_passed=34)))
