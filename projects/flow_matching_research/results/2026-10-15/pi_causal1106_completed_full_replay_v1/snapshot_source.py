"""Archive completed original training/point audit; native replay stays pending."""
from datetime import datetime, timezone
import json
from pathlib import Path
import psutil

from scripts.flow_matching import run_pi_causal1106_completed_full_replay_v1 as replay

root, read, digest = replay.ROOT, replay.read, replay.digest
runtime_path = 'configs/runtime/fm_pi_causal1106_completed_full_replay.v1.json'
runtime = read(runtime_path)
replay.verify(runtime)
case = runtime['cases'][0]
point = read(case['point_configuration'])
point_proof_path = point['output_directory'] + '/independent_full_recovery_completion.json'
proof = read(point_proof_path)
point_receipt_path = point['state_root'] + '/receipt.json'
receipt = read(point_receipt_path)
assert proof['passed'] and proof['original_id'] == replay.CANONICAL
assert proof['full_native_budget']['optimizer_minibatches'] == 1554
assert proof['scored_test_points'] == 87800 and proof['dropped_test_tail_points'] == 41
assert len(proof['original_full_score_shapes']) == 15
assert proof['all_posthoc_controls_recomputed'] == 15
assert proof['original_parameters'] == 1930289
assert not proof['full_native_checkpoint_reinference_passed']
assert not proof['strict_TAB_result'] and not proof['author_equivalence_certified']
assert receipt['exit_code'] == 0 and receipt['full_original_point_audit_passed']
assert receipt['resource_guard_aborted'] is False
assert digest(proof['result_path']) == proof['result_sha256']
assert digest(case['point_configuration']) == proof['config_sha256']
controller_path = runtime['state_root'] + '/status.json'
state = read(controller_path)
process = psutil.Process(state['pid'])
assert process.create_time() == state['create_time']
assert state['state'] == 'waiting_actual_full_native'
assert not state['verified_ready_original_ids']
native_path = runtime['configuration_root'] + '/' + replay.CANONICAL + '_native_ready.json'
native = read(native_path)
native_state_path = native['state_root'] + '/status.json'
native_state = read(native_state_path)
native_process = psutil.Process(native_state['pid'])
assert native_process.create_time() == native_state['create_time']
assert native_state['state'] == 'waiting_for_memory'
folder = root / 'projects/flow_matching_research/results/2026-10-15/pi_causal1106_completed_full_replay_v1'
snapshots = {'original_full_training_result.json': proof['result_path'],
             'completed_full_point_guarded_receipt.json': point_receipt_path}
for name, source in snapshots.items():
    with (folder / name).open('xb') as output:
        output.write((root / source).read_bytes())
snapshot = dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    canonical_id=replay.CANONICAL, original_scope=runtime['original_scope'],
    required_original_jobs=1140, required_original_training_seeds=[42,1103,1104,1105,1106,1107],
    selected_completed_original_models=1, additional_training_seed_slots=0,
    original_full_training_updates=1554, original_full_model_parameters=1930289,
    all_fifteen_saved_score_streams_and_posthoc_controls_audited=True,
    original_full_train_score_points=13238200, original_scored_test_points=87800,
    original_dropped_test_tail_points=41,
    independently_recomputed_saved_score_metrics=proof['independently_recomputed_metrics'],
    original_training_resource_receipt_available=False,
    full_native_checkpoint_reinference_passed=False,
    independent_completed_native_replay_passed=False,
    actual_live_controller=dict(pid=process.pid,create_time=process.create_time(),command=process.cmdline()),
    actual_live_native_admission_controller=dict(pid=native_process.pid,
        create_time=native_process.create_time(),command=native_process.cmdline()),
    native_state_at_capture=native_state,
    source_receipts=[dict(path=p,sha256=digest(p)) for p in [runtime_path,
        case['point_configuration'],point_proof_path,point_receipt_path,proof['result_path'],native_path]],
    author_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False)
with (folder/'current_pending_native_replay_snapshot.json').open('x',encoding='utf-8') as output:
    json.dump(snapshot,output,indent=2);output.write('\n')
(folder/'snapshot_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'launch_source.py').write_bytes((root/'tmp/launch_pi_causal1106_completed_full_replay_20261015.py').read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
(folder/'README.md').write_text(
    'Original causal-axis Pi-Transformer PSM seed1106 completed all 3 epochs / 1554 minibatches '
    'with 1,930,289 parameters. The guarded full saved-score audit independently checked '
    'all 15 streams and 15 posthoc controls, 13,238,200 overlapping training score points '
    'and 87,800 test points; the original 41-point dropped tail is retained.\n\n'
    'Saved-score PA F1: 0.9788686340. Pointwise no-PA F1 at the same source threshold: '
    '0.0191087853 (AUROC 0.4928185664; AP 0.2758567649). '
    'The source uses test data for early stopping and pooled train/test score calibration. '
    'These are protocol diagnostics, not strict TAB leaderboard results. '
    'The original training has no persisted resource-usage receipt; its guard history is unproven.\n\n'
    'Full checkpoint re-inference and the final independent replay review are still pending. '
    'The live native admission controller retains the original 24 GiB physical / 32 GiB commit '
    'admission, 16 GiB emergency headroom, CPU2 and unchanged score tolerances. '
    'The successful saved-score audit does not certify checkpoint re-inference. '
    '46 corrected-scope and original-stage validation checks passed; the separately preserved first validation has one erroneous frozen scope assertion and 45 passes.\n\n'
    'All 1140 original jobs, six seeds and 45-paper / 681-method / 273-data obligations remain. '
    'No retraining, extra seed, historical journal equivalence or between-seed uncertainty claim. '
    'The pinned paper, historical mirror, exact original configuration and dataset citations '
    'are carried in the runtime and child source receipts.\n',encoding='utf-8')
import shutil
import xml.etree.ElementTree as ET
corrected = root/'tmp/Pi_causal1106_corrected_scope_validation_20261015_v1.xml'
suites = list(ET.parse(corrected).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==46
assert all(int(x.get('errors',0))==int(x.get('failures',0))==0 for x in suites)
shutil.copyfile(corrected,folder/'corrected_scope_validation.xml')
shutil.copyfile(root/'tmp/Pi_causal1106_full_replay_registration_validation_20261015_v1.xml',folder/'preserved_pre_registration_validation_failure.xml')
shutil.copyfile(root/'tests/test_fm_pi_causal1106_scope_corrected_v2.py',folder/'independent_scope_correction_validation.py')
with (folder/'README.md').open('a',encoding='utf-8',newline='\n') as stream:
    stream.write('\nValidation limitation: the frozen v1 test file contains one accidental duplicate1106 in its expected six-seed list. The actual original queue remains[42,1103,1104,1105,1106,1107]. The initially failed XML is preserved; the independent corrected v2 validation passed all46 checks without edits to any frozen worker/configuration/source pins. The erroneous frozen assertion remains a documented test limitation, not a scientific or training failure.\n')
paths = ['scripts/flow_matching/run_pi_causal1106_completed_full_replay_v1.py',
         'scripts/flow_matching/register_pi_causal1106_completed_full_replay_v1.py',
         'tests/test_fm_pi_causal1106_completed_full_replay_v1.py',runtime_path]
paths += ['tests/test_fm_pi_causal1106_scope_corrected_v2.py']
paths += [p.relative_to(root).as_posix() for p in (root/runtime['configuration_root']).glob('*.json')]
paths += [p.relative_to(root).as_posix() for p in folder.rglob('*') if p.is_file()
          and p.suffix != '.log']
manifest = dict(previous_root='e8fa1422fedf2eeed94375722a70176fd34398b1',
    previous_parent='c253c6ddd3c0275d09430c745655345e63b12da6',paths=sorted(set(paths)),raw_evidence_whitespace_exceptions=['results/2026-10-15/pi_causal1106_completed_full_replay_v1/preserved_pre_registration_validation_failure.xml'])
(root/'tmp/pi_causal1106_full_point_and_native_admission_publication_manifest_20261015.json').write_text(
    json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(exact_files=len(manifest['paths']),point_audit_passed=True,
    full_checkpoint_replay_pending=True,actual_native_admission_pid=native_process.pid)))
