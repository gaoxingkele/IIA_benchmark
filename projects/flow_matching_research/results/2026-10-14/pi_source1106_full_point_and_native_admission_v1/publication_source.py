"""Publish complete original training/point audit and pending full native replay."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import psutil

from scripts.flow_matching import run_pi_source1106_completed_full_replay_v1 as replay

root, read, sha = replay.ROOT, replay.read, replay.digest
runtime_path = 'configs/runtime/fm_pi_source1106_completed_full_replay.v1.json'
runtime = read(runtime_path)
replay.verify(runtime)
point_path = runtime['cases'][0]['point_configuration']
point = read(point_path)
proof_path = point['output_directory'] + '/independent_full_recovery_completion.json'
proof = read(proof_path)
receipt_path = point['state_root'] + '/receipt.json'
receipt = read(receipt_path)
assert proof['passed'] and proof['original_id'] == replay.CANONICAL
assert proof['branch'] == 'source_runtime' and proof['seed'] == 1106
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
xml_path = 'tmp/Pi_source1106_full_replay_validation_20261014_v1.xml'
suites = list(ET.parse(root / xml_path).getroot().iter('testsuite'))
assert sum(int(s.get('tests', 0)) for s in suites) == 61
assert all(int(s.get('failures', 0)) == int(s.get('errors', 0)) == 0 for s in suites)

base = 'projects/flow_matching_research/results/2026-10-14/pi_source1106_full_point_and_native_admission_v1'
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
    validation_tests_passed=61, all_original_experiments_complete=False,
    historical_author_equivalence_certified=False, strict_TAB_result=False,
    inherited_review_template_prose_disclosure='The preserved final-review template boundary names source_runtime PSM seed42. Structured bindings, original seed1106 checkpoint/config/result hashes and full arrays identify the actual selected model. This pending registration does not certify the final review.',
    source_receipts=[dict(path=p,sha256=sha(p)) for p in (
        runtime_path,point_path,proof_path,receipt_path,native_path,proof['result_path'],xml_path)])
(folder / 'complete_training_and_point_audit_pending_native_replay.json').write_text(
    json.dumps(summary, indent=2) + '\n', encoding='utf-8')
(folder / 'publication_source.py').write_bytes(Path(__file__).read_bytes())
(folder / '.gitattributes').write_text('*.py -text\n*.json -text\n', encoding='utf-8')
(folder / 'README.md').write_text(
    'Original Pi source-runtime PSM seed1106 completed all3epochs/1554updates '
    'with1,930,289parameters. The separately guarded full saved-score audit passed: '
    'all15streams,13,238,200 overlapping training score points and87,800 covered test '
    'points were read and all15posthoc controls independently recomputed. '
    'Original41-point dropped test tail is retained.\n\n'
    'Independent source PA F1=0.9780571942; the same scores/threshold without PA give '
    'F1=0.0215759109,AUROC=0.4945081917,AP=0.2773853658. PA adds23,685 counted true '
    'positives while false positives remain687. Source test-based selection and pooled '
    'train/test calibration remain. This is local saved-score verification, not strict '
    'TAB or journal equivalence. Original training lacks a persisted guard receipt.\n\n'
    'The live three-stage pipeline advanced to complete native checkpoint inference. '
    'That stage and its final independent review remain pending in this snapshot; '
    'full32GiBcommit/24GiBphysical admission,16GiBemergency,CPU2 and original score '
    'tolerances are unchanged.61 relevant validation tests passed. Original1140Pi '
    'jobs/six seeds and45-paper/681-method/273-data scope remain required.\n\n'
    'The inherited frozen final-review template prose names source-runtime seed42; '
    'the actual canonical binding and all source hashes identify seed1106. Original '
    'templates are preserved. The complete native model replay is not yet certified.\n', encoding='utf-8')
registration_root = 'projects/flow_matching_research/results/2026-10-14/pi_source1106_completed_full_replay_v1'
shutil.copyfile(root / xml_path, root / registration_root / 'validation.xml')
targets = [base,registration_root,runtime['configuration_root'],runtime_path,
    'scripts/flow_matching/run_pi_source1106_completed_full_replay_v1.py',
    'scripts/flow_matching/register_pi_source1106_completed_full_replay_v1.py',
    'tests/test_fm_pi_source1106_completed_full_replay_v1.py']
paths = set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    raw = subprocess.check_output(['git',*args,'--',*targets],cwd=root)
    paths.update(p for p in raw.decode().splitlines() if not p.endswith('.log'))
assert paths and all(not p.endswith('.log') for p in paths)
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root).strip()
manifest = dict(previous_root='4df1a16120a4fc53a346371895007f4bba7bb7e7',
    previous_parent='a59b5734bf48bef360b04818f157d6f6cbae8851',paths=sorted(paths))
(root / 'tmp/Pi_source1106_full_point_native_admission_publication_manifest_20261014_v1.json').write_text(
    json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(exact_files=len(paths),original_training_and_full_point_audit_passed=True,
    full_native_model_replay_pending=True,live_controller_pid=process.pid,validation_tests_passed=61)))
