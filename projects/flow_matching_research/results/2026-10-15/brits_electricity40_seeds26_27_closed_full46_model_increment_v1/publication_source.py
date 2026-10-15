"""Publish two actual full native/independent completions with exact evidence."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent/'flow-matching-research'
PI = 'pi_author__source_runtime__psm__full__seed1107'
BRITS = 'saits_paper_brits_electricity_author_missing0.4_seed27'
PI_BASE = ROOT/'projects/flow_matching_research/results/2026-10-15/pi_source1107_actual_complete_model_replay_closure_v1'
BRITS_BASE = ROOT/'projects/flow_matching_research/results/2026-10-15/brits_electricity40_seeds26_27_closed_full46_model_increment_v1'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(65536), b''):
            digest.update(block)
    return digest.hexdigest()

xml = ROOT/'tmp/Pi_source1107_and_BRITS40_seed27_complete_full_replay_validation_20261015_v1.xml'
tree = ET.parse(xml).getroot()
suites = list(tree.iter('testsuite'))
assert sum(int(s.get('tests','0')) for s in suites) == 102
assert all(int(s.get(k,'0')) == 0 for s in suites for k in ('failures','errors','skipped'))
counts = Counter(case.attrib['classname'] for case in tree.iter('testcase'))
assert sum(v for k,v in counts.items() if '.test_fm_pi_' in k) == 62
assert sum(v for k,v in counts.items() if '.test_fm_pi_' not in k) == 40
runtime = read(ROOT/'configs/runtime/fm_pi_source1107_completed_full_replay.v1.json')
point = read(ROOT/runtime['cases'][0]['point_configuration'])
point_receipt_path = ROOT/point['state_root']/'receipt.json'
point_receipt = read(point_receipt_path)
assert point_receipt['exit_code'] == 0 and point_receipt['full_original_point_audit_passed']
assert not point_receipt['resource_guard_aborted']
shutil.copyfile(point_receipt_path, PI_BASE/'point_guarded_receipt.json')
pi = read(PI_BASE/'one_complete_original_model_and_all_fifteen_streams_closure.json')
assert pi['canonical_id'] == PI and pi['selected_complete_models'] == 1
assert pi['required_original_jobs'] == 1140
assert pi['required_original_training_seeds'] == [42,1103,1104,1105,1106,1107]
assert pi['complete_original_score_streams'] == 15 and pi['all_streams_bitwise_identical_to_original_saved_arrays']
assert pi['review_prose_bound_to_current_canonical_id'] and not pi['original_training_guard_verified']
assert not pi['strict_TAB_result'] and not pi['all_original_experiments_complete']
brits = read(BRITS_BASE/'two_complete_original_models_and_full_grid_reviews.json')
assert brits['complete_original_models'] == 2 and brits['fully_verified_original_seeds'] == [26,27]
assert brits['required_original_training_seeds'] == [26,27,28,29,30]
assert brits['reviewed_SAITS_BRITS_models'] == 46 and brits['required_SAITS_BRITS_models'] == 210
assert not brits['complete_five_seed_aggregate'] and not brits['strict_TAB_result']

for base in (PI_BASE, BRITS_BASE):
    shutil.copyfile(xml, base/'validation.xml')
    shutil.copyfile(__file__, base/'publication_source.py')
    (base/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
    with (base/'README.md').open('a',encoding='utf-8',newline='\n') as stream:
        stream.write('\n102 joint validation checks passed:62 Pi implementation/replay checks and40 '
                     'SAITS/BRITS implementation/guard/integrity checks. The new completions are exactly '
                     'Pi source-runtime PSM1107 and BRITS Electricity40 seed27; BRITS seed26 was '
                     'rechecked to retain the complete current2/5 cohort. No other model is promoted.\n')
    files = {p.relative_to(base).as_posix():sha(p) for p in base.rglob('*') if p.is_file()}
    record = dict(captured_utc=datetime.now(timezone.utc).isoformat(),joint_tests_passed=102,
                  relevant_Pi_tests=62,relevant_SAITS_BRITS_tests=40,
                  new_complete_original_models=[PI,BRITS],rechecked_BRITS_seeds=[26,27],
                  source_bytes_preserved=True,artifact_files_sha256=files,
                  additional_training_updates=0,additional_model_reinference_by_this_archive=0,
                  all_original_experiments_complete=False)
    (base/'archive_validation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    for relative, expected in files.items():
        assert sha(base/relative) == expected

targets = [PI_BASE.relative_to(ROOT).as_posix(),BRITS_BASE.relative_to(ROOT).as_posix(),
           runtime['configuration_root'],
           'projects/flow_matching_research/results/2026-10-15/pi_source1107_completed_full_replay_v1',
           'configs/reproducibility/fm_saits_completion_replay_queue_v1/'+BRITS+'.review.json',
           'projects/flow_matching_research/results/2026-10-11/saits_automatic_complete_model_reviews_v1/'+BRITS]
paths = set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    raw = subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True)
    paths.update(p for p in raw.splitlines() if not p.endswith('.log'))
git = lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
assert all(p.startswith('projects/flow_matching_research/') or p.startswith('configs/reproducibility/') for p in paths)
manifest = dict(previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(PARENT,'rev-parse','HEAD'),paths=sorted(paths))
target = ROOT/'tmp/Pi_source1107_and_BRITS40_seed27_closed_full46_publication_manifest_20261015_v1.json'
target.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(exact_files=len(paths),new_complete_original_models=[PI,BRITS],joint_tests=102,manifest=str(target))))
