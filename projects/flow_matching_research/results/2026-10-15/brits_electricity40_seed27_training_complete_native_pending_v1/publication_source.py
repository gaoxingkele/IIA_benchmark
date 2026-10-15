"""Freeze completed original training evidence; do not promote pending replay."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickletools
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import psutil

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = 'saits_paper_brits_electricity_author_missing0.4_seed27'
BASE = ROOT / 'projects/flow_matching_research/results/2026-10-15/brits_electricity40_seed27_training_complete_native_pending_v1'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            digest.update(block)
    return digest.hexdigest()

def scalars(path):
    wanted = {'next_epoch', 'finished', 'training_seconds', 'patience', 'train_step', 'training_step', 'epoch'}
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith('/data.pkl')]
        assert len(names) == 1
        raw = archive.read(names[0])
        assert len(raw) < 2_000_000
    values = {}
    operations = list(pickletools.genops(raw))
    for index, (op, value, _) in enumerate(operations):
        if op.name not in ('BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE') or value not in wanted:
            continue
        for following, number, _ in operations[index+1:index+5]:
            if following.name in ('MEMOIZE', 'BINPUT', 'LONG_BINPUT'):
                continue
            if following.name in ('BININT', 'BININT1', 'BININT2', 'LONG1', 'LONG4', 'BINFLOAT', 'FLOAT', 'NEWTRUE', 'NEWFALSE'):
                values[value] = True if following.name == 'NEWTRUE' else False if following.name == 'NEWFALSE' else number
            break
    return values

config_path = ROOT / 'configs/experiments/fm_saits_timeseries_jobs' / (IDENTITY + '.json')
config = read(config_path)
output = ROOT / config['output_root']
dataset = ROOT / config['dataset_root']
result_path = output / 'result.json'
result = read(result_path)
assert result['status'] == 'completed' and result['config'] == config
assert result['config_sha256'] == sha(config_path)
assert (config['method'], config['dataset'], config['seed'], config['missing_ratio']) == ('brits', 'electricity', 27, .4)
assert not config['diagnostic'] and result['parameters'] == 6993328
assert result['test_windows'] == 290 and result['target_count'] == 3537183 and result['bootstrap_groups'] == 10
assert sha(ROOT/config['author_config']) == config['author_config_sha256']
assert sha(dataset/'datasets.h5') == result['dataset_sha256'] == config['dataset_sha256']
resume, best = scalars(output/'resume.pt'), scalars(output/'best.pt')
assert resume['finished'] and resume['patience'] == 0
assert (resume['epoch'], resume['next_epoch'], resume['train_step']) == (905, 906, 6342)
assert resume['next_epoch'] * 7 == resume['train_step']
assert resume['training_seconds'] == result['training_seconds']
assert (best['epoch'], best['training_step']) == (874, 6125)
assert (result['validation_best']['epoch'], result['validation_best']['training_step']) == (874, 6125)

source_guard_path = Path('F:/aicoding/IIA_Data/experiments/saits_guarded_epoch271_full210_recovery_v2/receipts') / (IDENTITY+'.json')
guard = read(source_guard_path)
assert guard['exit_code'] == 0 and guard['full_result_validated'] and not guard['resource_guard_aborted']
try:
    retired = psutil.Process(32892)
    assert retired.create_time() != 1792026271.4812093
except psutil.NoSuchProcess:
    pass

native_path = ROOT / 'configs/reproducibility/fm_saits_completion_replay_queue_v1' / (IDENTITY+'.native.json')
review_path = native_path.with_name(IDENTITY+'.review.json')
native = read(native_path)
runtime_path = ROOT/'configs/runtime/fm_saits_completion_replay_queue.v1.json'
runtime = read(runtime_path)
assert not review_path.exists(), 'Review binding is generated only after a successful native replay'
for registered in (native,):
    for pin in registered['source_receipts']:
        assert sha(ROOT/pin['path']) == pin['sha256'], pin['path']
for pin in runtime['code_receipts']:
    assert sha(ROOT/pin['path']) == pin['sha256'], pin['path']
assert native['metric_tolerance'] == dict(rtol=2e-6, atol=2e-6)
assert native['settings']['minimum_commit_headroom_bytes'] == 28*1024**3
assert not (ROOT/native['artifact_directory']/'checkpoint_replay_audit.json').exists()
assert not (ROOT/runtime['review_output_root']/IDENTITY/'independent_full_replay_batch_verification.json').exists()

snapshots = {}
for label, path in {
    'actual_collector_state': ROOT/'experiments/runs/fm_saits_completion_replay_queue_v1/status.json',
    'actual_native_state': ROOT/native['state_root']/'status.json',
}.items():
    raw = path.read_bytes()
    snapshots[label] = (raw, json.loads(raw))
collector = snapshots['actual_collector_state'][1]
owner = psutil.Process(collector['pid'])
assert owner.create_time() == collector['create_time']
assert collector['required_original_models'] == 210 and len(collector['verified_original_ids']) == 45
assert IDENTITY not in collector['verified_original_ids']
assert collector['active_job'] == IDENTITY and collector['active_pid'] == 41852
state = snapshots['actual_native_state'][1]
pending = psutil.Process(41852)
assert pending.create_time() == state['create_time'] == 1792034404.410258
assert state['state'] == 'waiting_for_heavy_lock' and state.get('active_pid') is None
assert not state['shared_lock_owned']

xml = ROOT/'tmp/BRITS_electricity40_seed27_training_completion_validation_20261015_v1.xml'
suites = list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests', '0')) for s in suites) == 40
assert all(int(s.get(k, '0')) == 0 for s in suites for k in ('failures', 'errors', 'skipped'))
sources = dict(original_job_configuration=config_path, original_result=result_path,
               original_training_guard=source_guard_path, native_configuration=native_path,
               collector_runtime_configuration=runtime_path, original_data_audit=dataset/'data_audit.json')
artifact_paths = dict(original_job_configuration=config_path, original_result=result_path,
                     checkpoint=output/'best.pt', resume=output/'resume.pt', dataset=dataset/'datasets.h5',
                     original_group_statistics=output/'group_statistics.npz', data_audit=dataset/'data_audit.json')
summary = dict(captured_utc=datetime.now(timezone.utc).isoformat(), id=IDENTITY,
    full_original_training_and_test_completed=True, original_training_guard_passed=True,
    actual_outer_epochs=906, final_epoch_index=905, actual_training_steps=6342,
    original_maximum_epochs=10000, original_early_stop_patience=30, resume_scalar_metadata=resume,
    validation_best=result['validation_best'], parameters=6993328,
    original_full_test_metrics=result['metrics'], registered_full_test_shape=[290,100,370],
    source_test_windows=290, source_held_out_coordinates=3537183, source_month_groups=10,
    full_native_checkpoint_reinference_passed=False, separate_independent_full_grid_review_passed=False,
    complete_original_model_review_claimed=False, collector_fully_verified_original_models=45,
    independent_configuration_created=False,
    independent_configuration_policy='Generated by the existing collector only after successful full native replay',
    required_original_models=210, required_original_training_seeds=[26,27,28,29,30],
    complete_five_seed_aggregate=False, between_training_seed_mean_and_ci=None,
    source_within_model_month_block_bootstrap=result['month_block_bootstrap_95_ci'],
    source_prediction_bitwise_equivalence_claimed=False,
    full_artifact_sha256={key:sha(path) for key,path in artifact_paths.items()},
    source_receipts=[dict(name=key, path=str(path), sha256=sha(path)) for key,path in sources.items()],
    native_admission_settings=native['settings'], metric_tolerance=native['metric_tolerance'],
    pending_native_owner=dict(pid=pending.pid, create_time=pending.create_time(), exact_live=True,
                              actual_command=pending.cmdline(), state=state['state']),
    protocol_boundaries=result['protocol_boundaries'], citation=config['citation'], data_citation=config['data_citation'],
    validation_tests_passed=40, metadata_only_checkpoint_inspection=True,
    source_tensors_deserialized=False, additional_training_updates_in_this_archive=0,
    additional_model_reinference_in_this_archive=0, strict_TAB_result=False, author_equivalence_certified=False,
    original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
    full_goal_complete=False)
BASE.mkdir(parents=True, exist_ok=False)
for name, path in sources.items():
    shutil.copyfile(path, BASE/(name+'.json'))
for name, (raw, _) in snapshots.items():
    (BASE/(name+'.json')).write_bytes(raw)
(BASE/'complete_original_training_pending_full_replays.json').write_text(json.dumps(summary,indent=2)+'\n', encoding='utf-8')
shutil.copyfile(xml, BASE/'validation.xml')
shutil.copyfile(__file__, BASE/'publication_source.py')
(BASE/'.gitattributes').write_text('*.py -text\n*.json -text\n', encoding='utf-8')
(BASE/'README.md').write_text('''# BRITS / Electricity nominal40% / original seed27

The original model completed 906 outer epochs and 6342 training steps under the
unchanged maximum10000 / patience30 policy. Its source guard exited0 without a
resource abort. The checkpoint was selected on validation MAE at epoch874 /
step6125. The complete original test returned MAE1.030646800994873,
RMSE1.5363861322402954 and MRE0.5518696904182434 across290 windows,
3537183 held-out coordinates and10month groups.

Full native checkpoint inference is registered and waiting for the shared heavy
GPU lock. It has no successful terminal proof yet; the separate independent full
grid review is also pending. The collector remains45/210 fully reviewed models.
This snapshot does not add a fully reviewed model or a leaderboard result.

Frozen configuration, full dataset, checkpoint, resume, group-statistics and audit
hashes were checked. Checkpoint inspection read only scalar pickle opcodes; no
tensors were deserialized. All40 relevant implementation/guard/integrity tests
passed. Exact archived evidence bytes were checked after copying.

The registered full test grid is290x100x370; its complete native/independent
verification remains pending. Original within-model month-block intervals are
retained separately. No five-training-seed mean, STD or confidence interval is
claimed. Historical author preprocessing/tuning equivalence remains unproved;
this imputation result is not a strict TAB anomaly-detection result.

All original seeds26..30, all210 SAITS/BRITS models, and all45papers /
681method-baseline-ablation records /273data-task references remain in scope.
The original job is terminal and must not be trained again. Ongoing Pi native
inference and GRASP full training retain their original settings.
''', encoding='utf-8')
for name, path in sources.items():
    assert sha(BASE/(name+'.json')) == sha(path)
for name, (raw, _) in snapshots.items():
    assert (BASE/(name+'.json')).read_bytes() == raw
assert sha(BASE/'validation.xml') == sha(xml)

git = lambda cwd,*args: subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert not git(ROOT,'diff','--cached','--name-only')
paths = [p.relative_to(ROOT).as_posix() for p in BASE.iterdir() if p.is_file()]
paths += [native_path.relative_to(ROOT).as_posix()]
untracked = set(git(ROOT,'ls-files','--others','--exclude-standard','--',*paths).splitlines())
modified = set(git(ROOT,'diff','--name-only','--',*paths).splitlines())
paths = sorted(untracked | modified)
assert paths
manifest = dict(previous_root=git(ROOT,'rev-parse','HEAD'),
                previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=paths)
target = ROOT/'tmp/BRITS40_seed27_training_complete_native_pending_publication_manifest_20261015_v1.json'
target.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(training_and_test_complete=True, full_native_review_pending=True,
    separate_independent_review_pending=True, collector_fully_reviewed=45, required=210,
    tests_passed=40, exact_publication_files=len(paths), manifest=str(target))))
