"""Publish exact evidence; do not claim waiting model controls are completed."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime,timezone
import xml.etree.ElementTree as ET
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.flow_matching import prepare_original_full_entity_tsad_min105_v3 as prep
from scripts.flow_matching import verify_moment_masked_official_weights_v1 as moment
from scripts.flow_matching import run_full_smap_DCD105_min105_controls_v1 as dcd
read,sha,write=prep.read,prep.sha,prep.write
folder=ROOT/'projects/flow_matching_research/results/2026-10-16/full109_min105_actual_data_and_corrected873_worker_routing_v1'
assert folder.is_dir()


def copy(source,name):
    target=folder/name;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        assert sha(target)==sha(source)
        return
    shutil.copyfile(source,target)
    assert sha(target)==sha(source)


pc=read(ROOT/prep.CONFIG);prep.verify(pc)
report=Path(pc['artifact_root'])/prep.REPORT;proof=read(report)
guard_path=ROOT/pc['state_root']/'guard_receipt.json';guard=read(guard_path)
assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['proof_sha256']==sha(report)
assert proof['passed'] and proof['configuration_sha256']==sha(ROOT/prep.CONFIG)
assert proof['entities']==109 and proof['datasets']==3 and proof['CSV_files_compared']==191
assert proof['full_input_coordinate_reconstruction'] and proof['complete_serialized_arrays_verified']
assert proof['cross_version']['unchanged_entities']==107 and set(proof['cross_version']['changed_entities'])==set(prep.CHANGED)
assert proof['cross_version']['all_test_label_overlays_verified']==298
assert sha(proof['prepared_manifest'])==proof['prepared_manifest_sha256']
copy(report,'actual_full109_data_and_cross_version_proof.json')
copy(guard_path,'actual_full109_data_guard.json')
copy(proof['prepared_manifest'],'actual_full109_common105_manifest.json')

routing=Path('F:/aicoding/IIA_Data/experiments/original_full873_actual_window_geometry_routing_20261016_v2.json')
rv=read(routing)
assert rv['all_source_cases']==873 and rv['actual_worker_predicted_incompatible_cases']==5 and rv['corrected_MOMENT_false_positive_cases']==6
copy(routing,'actual_all873_worker_routing_geometry_v2.json')
validation=Path('F:/aicoding/IIA_Data/experiments/full109_min105_MOMENT_masked_DCD105_all111_tests_20261016.xml')
suites=list(ET.parse(validation).getroot().iter('testsuite'))
assert sum(int(s.get('tests','0')) for s in suites)==111
assert all(int(s.get(k,'0'))==0 for s in suites for k in ['errors','failures','skipped'])
copy(validation,'validation111.xml')

paths=[prep.MODULE.replace('.','/')+'.py',prep.CONFIG,prep.TEST,
    moment.MODULE.replace('.','/')+'.py',moment.CONFIG,moment.TEST,
    dcd.MODULE.replace('.','/')+'.py',dcd.CONFIG,dcd.TEST,
    'scripts/flow_matching/run_moment_masked_context_job_v1.py',
    'scripts/flow_matching/audit_full873_window_geometry_routing_v2.py',
    'src/iia_benchmark/models/fm_moment_masked_context_v1.py',
    'tests/test_fm_moment_masked_context_v1.py',
    'tests/test_fm_full873_window_geometry_routing_v2.py']
paths += [moment.model_path(size) for size in moment.SIZES]
source_map=[]
for index,path in enumerate(paths):
    name='frozen_sources/'+f'{index:02d}_'+Path(path).stem[:48]+Path(path).suffix
    # Recover only generated partial publication copies after Windows MAX_PATH;
    # both absolute paths must stay inside this exact new evidence directory.
    previous=folder/'frozen_sources'/path;target=folder/name
    if previous.exists():
        assert previous.resolve().is_relative_to(folder.resolve()) and target.resolve().is_relative_to(folder.resolve())
        assert not target.exists() and sha(previous)==sha(ROOT/path)
        previous.rename(target)
    copy(ROOT/path,name)
    source_map.append(dict(source=path,archive=name,sha256=sha(ROOT/path)))
write(folder/'frozen_source_mapping.json',dict(records=source_map))

states=[]
for runner in [moment,dcd]:
    c=read(ROOT/runner.CONFIG);runner.verify(c)
    launch=read(ROOT/c['state_root']/'launch.json');state=read(ROOT/c['state_root']/'status.json')
    owner=psutil.Process(launch['pid'])
    assert owner.create_time()==launch['create_time']==state['create_time'] and owner.cmdline()==launch['command']
    errors=ROOT/c['state_root']/'controller.stderr.log'
    assert not errors.read_bytes()
    states.append(dict(configuration=runner.CONFIG,actual_owner=dict(pid=owner.pid,create_time=owner.create_time(),command=owner.cmdline()),
        actual_status_snapshot=state,settings=c['settings'],resource_lock=c['resource_lock'],
        actual_source_nn_jobs_complete=0,waiting_is_not_failure_or_completion=True))
write(folder/'actual_pending_control_handles_not_results.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),records=states,
    new_real_dataset_performance_results_claimed=0,official_weight_checks_completed=False,all_original_experiments_complete=False))

failed=Path('F:/aicoding/IIA_Data/experiments/full_smap_DCD105_min105_controls_v1_tests_20261016.xml')
failed_suites=list(ET.parse(failed).getroot().iter('testsuite'))
assert sum(int(s.get('failures','0')) for s in failed_suites)==1
oldconfig=Path('F:/aicoding/IIA_Data/experiments/full_smap_DCD105_min105_prelaunch_validation_v1/configuration_before_test_only_correction.json')
write(folder/'preserved_prelaunch_test_failure.json',dict(
    failure_xml=dict(path=str(failed),sha256=sha(failed)),configuration_before_test_fix=dict(path=str(oldconfig),sha256=sha(oldconfig)),
    cause='New test wrongly looked for nested epochs100;actual frozen local source config has outer epochs3',
    correction='Test now verifies actual original outer epochs3;only new prelaunch test receipt refreshed',
    original_model_or_scientific_job_changed=False,scientific_job_retry_count=0,
    local3epochs_not_original_paper_training_certification=True))

git=lambda cwd,*a:subprocess.check_output(['git','-C',str(cwd),*a],text=True).strip()
copy(__file__,'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),tests_passed=111,
    actual109entity_data_preparation_completed=True,actual873routes_checked=True,
    DCD105_predicted_incompatible_cases=5,MOMENT_generic_gate_false_positives_corrected=6,
    new_real_dataset_model_results_claimed=0,all_original_experiments_complete=False,
    files_sha256={p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}))
ara=ROOT/'projects/flow_matching_research/ara/execution/full109_min105_and_actual873_worker_routing.2026-10-16_v1.json'
write(ara,dict(evidence=folder.relative_to(ROOT).as_posix(),actual_full109_preparation=True,tests_passed=111,
    actual_worker_routing_cases=873,corrected_MOMENT_false_positive_cases=6,DCD105_true_geometry_conflicts=5,
    registered_new_DCD105_controls=5,registered_official_MOMENT_sizes=3,
    official_weight_checks_completed=False,new_real_dataset_performance_results_claimed=0,
    original_scope=dict(papers=45,method_records=681,data_tasks=273,paper_scalar_targets=702),all_original_experiments_complete=False))
paths += [ara.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()]
readme=folder/'README.md'
assert readme.exists()
paths=sorted(set(paths));assert not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/full109_min105_actual_routing_publication_manifest_20261016_v1.json',
    dict(previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=paths))
print(json.dumps(dict(exact_files=len(paths),tests_passed=111,actual_full109_preparation=True,
    actual_nn_controls_complete=0,live_owners=[r['actual_owner']['pid'] for r in states],all_original_experiments_complete=False)))
