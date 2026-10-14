"""Publish one complete Transformer baseline with the full38-model ledger."""
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

root=Path(__file__).resolve().parents[1]
base='projects/flow_matching_research/results/2026-10-14/grasp_transformer_seed1106_full38_model_increment_v1'
folder=root/base
closure=json.loads((folder/'actual_full38_model_increment_closure.json').read_text())
assert closure['complete_original_training_and_full_native_replay_and_independent_recipe_review']
assert closure['actual_current_unique_complete_models']==38 and closure['parameter_count']==220338
assert closure['epochs']==1500 and closure['optimizer_updates']==93000 and len(closure['recipes'])==1
assert closure['complete_transformer_Swan_seeds']==[1103,1104,1105,1106]
assert closure['required_original_entity_models']==9120 and not closure['full_ten_seed_transformer_cohort_complete']
assert not closure['all_original_experiments_complete'] and not closure['strict_TAB_result']
xml=root/'tmp/GRASP_transformer1106_full38_validation_20261014_v1.xml'
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
manifest=dict(previous_root='5d4c6d6ef2987bbfd3cf694c03a85d01b921b2df',
    previous_parent='f06dd4476c0006f650c35d79dbdab464f7992f77',paths=sorted(paths))
(root/'tmp/GRASP_transformer1106_full38_publication_manifest_20261014_v1.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(exact_files=len(paths),complete_original_models=38,required_original_models=9120,
    original_transformer_SWAN_seeds_closed=[1103,1104,1105,1106],validation_tests_passed=34)))
