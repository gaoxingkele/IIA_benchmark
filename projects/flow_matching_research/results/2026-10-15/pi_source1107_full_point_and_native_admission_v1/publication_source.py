"""Publish complete original training/point audit and pending full native replay."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import psutil

from scripts.flow_matching import run_pi_source1107_completed_full_replay_v1 as replay

root, read, sha = replay.ROOT, replay.read, replay.digest
runtime_path = 'configs/runtime/fm_pi_source1107_completed_full_replay.v1.json'
runtime = read(runtime_path)
replay.verify(runtime)
point_path = runtime['cases'][0]['point_configuration']
point = read(point_path)
proof_path = point['output_directory'] + '/independent_full_recovery_completion.json'
proof = read(proof_path)
receipt_path = point['state_root'] + '/receipt.json'
receipt = read(receipt_path)
assert proof['passed'] and proof['original_id'] == replay.CANONICAL
assert proof['branch'] == 'source_runtime' and proof['seed'] == 1107
assert proof['config_sha256'] == sha(point_path)
assert proof['result_sha256'] == sha(proof['result_path'])
assert proof['full_native_budget']['optimizer_minibatches'] == 1554
assert len(proof['original_full_score_shapes']) == 15
assert proof['original_full_score_shapes']['train_energy'] == [13238200]
assert proof['original_full_score_shapes']['labels'] == [87800]
assert proof['all_posthoc_controls_recomputed'] == 15
assert receipt['exit_code'] == 0 and receipt['full_original_point_audit_passed']
assert not receipt['resource_guard_aborted']
assert proof['resource_usage']['original_training_guard_verified'] is False
assert not proof['full_native_checkpoint_reinference_passed']
assert not proof['strict_TAB_result'] and not proof['author_equivalence_certified']
for pin in point['source_receipts']:
    assert sha(pin['path']) == pin['sha256']
state = read(runtime['state_root'] + '/status.json')
process = psutil.Process(state['pid'])
assert process.create_time() == state['create_time']
child = psutil.Process(state['active_pid'])
assert child.create_time() == state['active_create_time']
assert child.cmdline() == state['actual_command']
assert state['state'] == 'waiting_actual_full_native'
native_path = runtime['configuration_root'] + '/' + replay.CANONICAL + '_native_ready.json'
native = read(native_path)
native_state = read(native['state_root'] + '/status.json')
xml_path = 'tmp/Pi_source1107_completed_full_replay_validation_20261015_v1.xml'
suites = list(ET.parse(root / xml_path).getroot().iter('testsuite'))
assert sum(int(s.get('tests', 0)) for s in suites) == 51
assert all(int(s.get('failures', 0)) == int(s.get('errors', 0)) == 0 for s in suites)

review=read(runtime['review_template'])
original_review=read('configs/reproducibility/fm_pi_original_completed_full_replay_review.2026-10-11_v1.json')
assert replay.CANONICAL in review['boundary']
assert {k:v for k,v in review.items() if k!='boundary'}=={k:v for k,v in original_review.items() if k!='boundary'}

base = 'projects/flow_matching_research/results/2026-10-15/pi_source1107_full_point_and_native_admission_v1'
folder = root / base
folder.mkdir(exist_ok=False)
for path, name in [(proof['result_path'], 'original_complete_training_result.json'),
                   (receipt_path, 'full_point_guarded_receipt.json'),
                   (xml_path, 'validation.xml')]:
    shutil.copyfile(root / path, folder / name)
summary = dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    canonical_id=replay.CANONICAL, original_scope=runtime['original_scope'],
    original_required_jobs=1140, original_required_training_seeds=[42,1103,1104,1105,1106,1107],
    complete_original_training_epochs=3, optimizer_minibatches=1554, parameters=1930289,
    full_original_saved_score_points_train=13238200, scored_test_points=87800,
    original_dropped_test_tail_points=41, complete_saved_score_streams_audited=15,
    independently_recomputed_metrics=proof['independently_recomputed_metrics'],
    confusion_no_PA=proof['confusion_no_PA'], confusion_author_PA=proof['confusion_author_PA'],
    source_threshold=proof['source_threshold'],
    full_point_audit_guarded_passed=True, original_training_guard_verified=False,
    full_native_checkpoint_reinference_passed=False, full_final_independent_review_passed=False,
    live_pipeline=dict(pid=process.pid, create_time=process.create_time(),
        phase=state['state'], active_pid=child.pid, active_create_time=child.create_time(),
        actual_command=child.cmdline(), native_state=native_state),
    additional_training_seed_slots=0, additional_optimizer_updates=0,
    validation_tests_passed=51, all_original_experiments_complete=False,
    historical_author_equivalence_certified=False, strict_TAB_result=False,
    review_template_metadata_alignment_verified=True,
    source_receipts=[dict(path=p,sha256=sha(p)) for p in (
        runtime_path,point_path,proof_path,receipt_path,native_path,proof['result_path'],xml_path)])
(folder / 'complete_training_and_point_audit_pending_native_replay.json').write_text(
    json.dumps(summary, indent=2) + '\n', encoding='utf-8')
(folder / 'publication_source.py').write_bytes(Path(__file__).read_bytes())
(folder / '.gitattributes').write_text('*.py -text\n*.json -text\n', encoding='utf-8')
(folder / 'README.md').write_text(
    'Original Pi source-runtime PSM seed1107 completed all3epochs/1554updates with1930289parameters. '
    'The guarded complete saved-score audit independently checked all15streams,13238200 overlapping '
    'training score points and87800 test points. The original41-point dropped test tail is retained.\n\n'
    'Actual saved-score metrics: '+json.dumps(proof['independently_recomputed_metrics'])+'\n\n'
    'Complete checkpoint re-inference and its final independent full-array review remain pending. '
    'The actual live controller and native admission state are archived in the JSON summary. '
    'Original32GiBcommit/24GiBphysical admission,16GiBemergency,CPU2 and score tolerances remain. '
    'All51 relevant checks passed. Original1140jobs/all six seeds and45/681/273scope remain required.\n\n'
    'Original training has no persisted resource-guard receipt. Source test-based model selection, '
    'pooled train/test calibration and point adjustment remain explicitly separated from unadjusted '
    'controls. No author-equivalence or strictTAB certificate is issued. The new readonly review '
    'template aligns its prose with the actual source1107 identity before freezing; every other '
    'field is identical to the original template. Original frozen files were preserved.\n',encoding='utf-8')
registration_root = 'projects/flow_matching_research/results/2026-10-15/pi_source1107_completed_full_replay_v1'
shutil.copyfile(root / xml_path, root / registration_root / 'validation.xml')
shutil.copyfile(root/'tmp/launch_pi_source1107_completed_full_replay_20261015.py',root/registration_root/'launch_source.py')
(root/registration_root/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
targets = [base,registration_root,runtime['configuration_root'],runtime_path,
    'scripts/flow_matching/run_pi_source1107_completed_full_replay_v1.py',
    'scripts/flow_matching/register_pi_source1107_completed_full_replay_v1.py',
    'tests/test_fm_pi_source1107_completed_full_replay_v1.py']
paths = set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    raw = subprocess.check_output(['git',*args,'--',*targets],cwd=root)
    paths.update(p for p in raw.decode().splitlines() if not p.endswith('.log'))
assert paths and all(not p.endswith('.log') for p in paths)
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root).strip()
manifest = dict(previous_root='003d351b426b6a84ff79161734db06f326e80b6e',
    previous_parent='b43517048c7b678c6d7d20e877c20d213c4ce27c',paths=sorted(paths))
(root / 'tmp/Pi_source1107_full_point_native_admission_publication_manifest_20261014_v1.json').write_text(
    json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(exact_files=len(paths),original_training_and_full_point_audit_passed=True,
    full_native_model_replay_pending=True,live_controller_pid=process.pid,validation_tests_passed=51)))
