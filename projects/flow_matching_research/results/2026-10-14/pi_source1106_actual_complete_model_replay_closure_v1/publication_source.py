"""Publish the complete original source PSM1106 checkpoint/array closure."""
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from scripts.flow_matching import run_pi_source1106_completed_full_replay_v1 as replay

root=replay.ROOT
runtime=replay.read('configs/runtime/fm_pi_source1106_completed_full_replay.v1.json')
replay.verify(runtime)
base='projects/flow_matching_research/results/2026-10-14/pi_source1106_actual_complete_model_replay_closure_v1'
folder=root/base
closure=replay.read(base+'/one_complete_original_model_and_all_fifteen_streams_closure.json')
assert closure['canonical_id']==replay.CANONICAL and closure['selected_complete_models']==1
assert closure['all_streams_bitwise_identical_to_original_saved_arrays']
assert closure['complete_original_score_streams']==15
assert closure['full_train_score_points']==13238200 and closure['scored_original_test_points']==87800
assert closure['original_dropped_tail_points']==41 and closure['optimizer_minibatches']==1554
assert closure['original_training_guard_verified'] is False
assert closure['inherited_proof_prose_mismatch_disclosure']
assert not closure['all_original_experiments_complete'] and not closure['strict_TAB_result']
xml=root/'tmp/Pi_source1106_complete_native_replay_closure_validation_20261014_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==61
assert all(int(s.get('failures',0))==int(s.get('errors',0))==0 for s in suites)
shutil.copyfile(xml,folder/'validation.xml')
(folder/'publication_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
with (folder/'README.md').open('a',encoding='utf-8',newline='\n') as stream:
    stream.write('\n61 relevant validation checks passed. The original training has no persisted '
        'resource receipt; its guard history is unproven. The separate complete point, native '
        'and final independent audit stages each passed their original guards. Frozen final '
        'review prose names source-runtime seed42; actual structured canonical identity, '
        'seed1106 and all source/checkpoint/config/result/array digests identify the completed '
        'source-runtime seed1106. The frozen proof is preserved unchanged.\n')
pending='projects/flow_matching_research/results/2026-10-14/pi_source1106_full_point_and_native_admission_v1'
with (root/pending/'README.md').open('a',encoding='utf-8',newline='\n') as stream:
    stream.write('\nCompletion update: all15streams now have successful full native re-inference '
        'and independent review; see ../pi_source1106_actual_complete_model_replay_closure_v1/README.md. '
        'The preceding pending snapshot and frozen configuration evidence are preserved.\n')
targets=[base,pending+'/README.md',runtime['configuration_root'],
    'projects/flow_matching_research/results/2026-10-14/pi_source1106_completed_full_replay_v1']
paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    raw=subprocess.check_output(['git',*args,'--',*targets],cwd=root)
    paths.update(p for p in raw.decode().splitlines() if not p.endswith('.log'))
assert paths and not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root).strip()
manifest=dict(previous_root='e9f2b69ada1a5b6bfd64a4f695130ab8cc651688',
    previous_parent='cd25b8e178705d7b40a31633f6d95b7378b4e1d6',paths=sorted(paths))
(root/'tmp/Pi_source1106_complete_model_publication_manifest_20261014_v1.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(exact_files=len(paths),complete_original_models=1,bitwise_complete_streams=15,
    validation_tests_passed=61,full_original_scope_retained=True)))
