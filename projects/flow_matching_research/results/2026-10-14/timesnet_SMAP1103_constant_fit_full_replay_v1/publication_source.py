"""Publish completed original local training and two live full-audit queues."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import psutil

from scripts.flow_matching import audit_timesnet_constant_fit_complete_v1 as audit

root=audit.ROOT
base='projects/flow_matching_research/results/2026-10-14/timesnet_SMAP1103_constant_fit_full_replay_v1'
folder=root/base
reg=audit.read(folder/'registration.json')
states={}
for mode in ('point','native'):
    config=audit.read(root/reg['configurations'][mode]);queue,job,result,model=audit.binding(config)
    launch=audit.read(folder/('actual_'+mode+'_controller_launch.json'))
    process=psutil.Process(launch['pid'])
    assert process.create_time()==launch['create_time'] and process.cmdline()==launch['actual_command']
    state=audit.read(root/config['state_root']/'status.json')
    assert state['pid']==process.pid and state['create_time']==process.create_time()
    assert state['active_pid'] is None and state['state'] in ('waiting_for_memory','waiting_for_resource_lock')
    states[mode]=dict(pid=process.pid,create_time=process.create_time(),actual_command=process.cmdline(),state=state)
xml=root/'tmp/TimesNet_SMAP1103_constant_fit_complete_validation_20261014_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==37
assert all(int(s.get('failures',0))==int(s.get('errors',0))==0 for s in suites)
guard=audit.read(root/config['terminal_receipt'])
assert guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['full_registered_budget_retained']
assert guard['job_id']==audit.CANONICAL
for source,name in [(root/job['output_directory']/'result.json','original_complete_training_result.json'),
    (root/config['terminal_receipt'],'original_successful_guarded_training_receipt.json'),
    (root/'tmp/launch_timesnet_SMAP1103_native_replay_20261014.py','native_launch_source.py')]:
    shutil.copyfile(source,folder/name)
manifest=audit.read(root/job['dataset_manifest_path']);dataset=next(d for d in manifest['datasets'] if d['id']=='smap')
summary=dict(captured_utc=datetime.now(timezone.utc).isoformat(),canonical_id=audit.CANONICAL,
    original_scope=audit.SCOPE,full_original_baseline_queue_jobs=35,full_corrected_data_controls=291,
    original_full_registered_training_and_scoring_completed=True,source_guard_exited_without_abort=True,
    original_training_seconds=result['training_seconds'],original_scoring_seconds=result['score_seconds'],
    configured_training_epochs=3,persisted_epoch_update_counter_available=False,full_model_parameters=28132633,
    original_full_entities=len(dataset['entities']),original_full_test_points=sum(e['test_points'] for e in dataset['entities']),
    original_source_checkpoint_sha256=result['checkpoint_sha256'],original_source_scores_sha256=result['scores_sha256'],
    source_primary_metrics_not_yet_independently_verified=result['metrics']['strict']['1']['micro'],
    source_conditional_entity_cluster_ci95=result['metrics']['strict']['1']['f1_ci95'],
    source_event_metrics=result['metrics']['strict']['1']['event_metrics'],
    source_test_calibrated_core_diagnostic=result['metrics']['tab_core_diagnostic'],
    source_evaluation=queue['evaluation'],point_and_native_actual_controller_states=states,
    full_saved_score_independent_review_passed=False,full_native_checkpoint_reinference_passed=False,
    full_TAB_range_metrics_completed=False,validation_tests_passed=37,
    original_required_seed_cohort=[1103,1104,1105,1106,1107],complete_five_seed_aggregate=False,
    no_new_optimizer_updates=True,additional_training_seed_slots=0,
    historical_author_equivalence_certified=False,full_TAB_training_harness_certified=False,
    all_original_experiments_complete=False,
    source_receipts=[dict(path=p,sha256=audit.sha(root/p)) for p in [
        *reg['configurations'].values(),job['output_directory']+'/result.json',config['terminal_receipt'],
        job['model_config'],job['dataset_manifest_path'],base+'/validation.xml']])
(folder/'completed_original_training_with_pending_full_audits.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
(folder/'publication_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
(folder/'README.md').write_text(
    'TimesNet local registered SMAP constant-fit preprocessing control seed1103 '
    'completed its original configured3epochs and all original train/validation/test scoring '
    'across51entities/408,160test points. The full28,132,633parameter model,CPU2,batch128 '
    'and original capacity/optimizer defaults remain. Original successful resource receipt '
    'hasexit0/noabort/full_registered_budget_retained. A per-epoch optimizer counter was '
    'not recorded; configured budgets are not substituted for a measured update trace.\n\n'
    'Source validation1% noPA F1=0.0453082580,AUROC=0.4362539773,AP=0.1213720771. '
    'Conditional entity-cluster F1 CI=[0.0104659048,0.1044334680],200resamples. '
    'This is uncertainty conditional on one model, not a five-seed algorithm comparison. '
    'Source pooled train/test calibration and best test-label threshold-grid diagnostics '
    'give plain F1max=0.1451872366 and PA F1max=0.7774734352; these are separate oracle '
    'diagnostics and are not strict leaderboard metrics.\n\n'
    'Full independent saved-score/point/core/event/bootstrap verification and checkpoint '
    're-inference are registered in two actual live controllers using the same existing '
    'serial readonly lock. Both currently await resource admission. Full native score '
    'agreement must be bitwise at the original environment/precision; no new fitting is '
    'allowed.37 relevant checks passed. Neither actual audit has passed in this snapshot.\n\n'
    'All35original baseline slots,291corrected-data controls,all five original seeds '
    '[1103,1104,1105,1106,1107] and45-paper/681-method/273-data scope remain. '
    'Full TAB range evaluation remains with the pre-existing291-job collector. '
    'This local transcribed implementation is not a historical author-harness or full '
    'TAB training-harness reproduction. Literature: https://doi.org/10.48550/arXiv.2210.02186; '
    'data/protocol: https://github.com/decisionintelligence/TAB.\n',encoding='utf-8')
targets=[base,*reg['configurations'].values(),'scripts/flow_matching/audit_timesnet_constant_fit_complete_v1.py',
    'tests/test_fm_timesnet_constant_fit_complete_v1.py']
paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    raw=subprocess.check_output(['git',*args,'--',*targets],cwd=root)
    paths.update(p for p in raw.decode().splitlines() if not p.endswith('.log'))
assert paths and not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root).strip()
publication=dict(previous_root='e2d7f86e80cfad052bc9dea0be1b78c15cee730f',
    previous_parent='8b8a58bc4bbf49c47a85ef6254ff62ececa98c6a',paths=sorted(paths))
(root/'tmp/TimesNet_SMAP1103_complete_training_pending_replay_publication_manifest_20261014_v1.json').write_text(
    json.dumps(publication,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(exact_files=len(paths),original_training_completed=True,full_native_replay_pending=True,
    point_controller=states['point']['pid'],native_controller=states['native']['pid'],validation_tests_passed=37)))
