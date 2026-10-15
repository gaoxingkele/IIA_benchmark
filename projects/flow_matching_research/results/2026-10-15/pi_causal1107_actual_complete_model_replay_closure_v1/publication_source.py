"""Publish only the actual completed causal PSM1107 model and its full proofs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent / 'flow-matching-research'
ID = 'pi_author__causal_axis_repaired__psm__full__seed1107'
BASE = ROOT / 'projects/flow_matching_research/results/2026-10-15/pi_causal1107_actual_complete_model_replay_closure_v1'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

xml = ROOT / 'tmp/Pi_causal1107_actual_completed_full_replay_validation_20261015_v1.xml'
suites = list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests', 0)) for s in suites) == 82
assert all(int(s.get(k, 0)) == 0 for s in suites for k in ('failures', 'errors', 'skipped'))
runtime = read(ROOT / 'configs/runtime/fm_pi_causal1107_completed_full_replay.v1.json')
point = read(ROOT / runtime['cases'][0]['point_configuration'])
point_receipt_path = ROOT / point['state_root'] / 'receipt.json'
point_receipt = read(point_receipt_path)
assert point_receipt['exit_code'] == 0 and point_receipt['full_original_point_audit_passed']
assert not point_receipt['resource_guard_aborted']
closure = read(BASE / 'one_complete_original_model_and_all_fifteen_streams_closure.json')
assert closure['canonical_id'] == ID and closure['selected_complete_models'] == 1
assert closure['required_original_jobs'] == 1140
assert closure['required_original_training_seeds'] == [42, 1103, 1104, 1105, 1106, 1107]
assert closure['complete_original_score_streams'] == 15
assert closure['all_streams_bitwise_identical_to_original_saved_arrays']
assert closure['full_native_checkpoint_reinference_and_independent_stream_metric_review_complete']
assert closure['review_prose_bound_to_current_canonical_id']
assert not closure['original_training_guard_verified']
assert not closure['strict_TAB_result'] and not closure['all_original_experiments_complete']
deferred_root = ROOT / 'experiments/runs/fm_pi_causal1107_deferred_full_replay_v2'
automatic = read(deferred_root / 'receipt.json')
assert automatic['exit_code'] == 0 and automatic['requested_case_full_replay_pipeline_passed']
assert not automatic['original_training_restarted'] and automatic['added_training_seed_slots'] == 0
for source, target in [(xml, 'validation.xml'), (point_receipt_path, 'point_guarded_receipt.json'),
                       (deferred_root / 'receipt.json', 'automatic_deferred_completion_receipt.json')]:
    shutil.copyfile(source, BASE / target)
shutil.copyfile(__file__, BASE / 'publication_source.py')
(BASE / '.gitattributes').write_text('*.py -text\n*.json -text\n', encoding='utf-8')
files = {p.relative_to(BASE).as_posix(): sha(p) for p in BASE.rglob('*') if p.is_file()}
record = dict(captured_utc=datetime.now(timezone.utc).isoformat(), relevant_tests_passed=82,
              new_complete_original_models=[ID], source_bytes_preserved=True,
              artifact_files_sha256=files, additional_training_updates=0,
              additional_model_reinference_by_this_archive=0, all_original_experiments_complete=False)
(BASE / 'archive_validation.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
for relative, expected in files.items():
    assert sha(BASE / relative) == expected
targets = [BASE.relative_to(ROOT).as_posix(), runtime['configuration_root'],
           'projects/flow_matching_research/results/2026-10-15/pi_causal1107_completed_full_replay_v1']
paths = set()
for args in [('diff', '--name-only'), ('ls-files', '--others', '--exclude-standard')]:
    raw = subprocess.check_output(['git', *args, '--', *targets], cwd=ROOT, text=True)
    paths.update(p for p in raw.splitlines() if not p.endswith('.log'))
git = lambda cwd, *args: subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()
assert paths and not git(ROOT, 'diff', '--cached', '--name-only')
assert all(p.startswith('projects/flow_matching_research/') or p.startswith('configs/reproducibility/') for p in paths)
manifest = dict(previous_root=git(ROOT, 'rev-parse', 'HEAD'),
                previous_parent=git(PARENT, 'rev-parse', 'HEAD'), paths=sorted(paths))
target = ROOT / 'tmp/Pi_causal1107_closed_full_model_publication_manifest_20261015_v1.json'
target.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(exact_files=len(paths), new_complete_original_model=ID, tests_passed=82)))
