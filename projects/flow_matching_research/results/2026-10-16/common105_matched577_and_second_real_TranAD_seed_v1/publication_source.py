import csv,io,json,shutil,subprocess,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import xml.etree.ElementTree as ET
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.flow_matching import run_common105_matched577_v1 as s
from scripts.flow_matching import run_TAB_full878_metric_suite_v1 as tab
folder=ROOT/'projects/flow_matching_research/results/2026-10-16/common105_matched577_and_second_real_TranAD_seed_v1'
assert {p.name for p in folder.iterdir()}=={'README.md'}

def copy(source,name):
    target=folder/name;assert not target.exists();shutil.copyfile(source,target);assert s.sha(source)==s.sha(target)

c=s.read(ROOT/s.CONFIG);s.verify(c);material=s.verify_materialization(c)
assert len(material['cohort'])==873 and sum(q['jobs'] for q in material['queues'])==577
copy(Path(c['artifact_root'])/'materialization.json','actual577queues_and873_cohort_materialization.json')
validation=Path('F:/aicoding/IIA_Data/experiments/common105_matched577_full873_tests_20261016_v1.xml')
xs=list(ET.parse(validation).getroot().iter('testsuite'));assert sum(int(x.get('tests','0')) for x in xs)==28
assert all(int(x.get(k,'0'))==0 for x in xs for k in ['failures','errors','skipped'])
copy(validation,'validation28.xml')
paths=[s.MODULE.replace('.','/')+'.py',s.CONFIG,s.TEST]
for i,p in enumerate(paths):copy(ROOT/p,f'original{i:02d}'+Path(p).suffix)
state=s.read(ROOT/c['state_root']/'status.json');handles=[]
for h in state['family_handles']:
    p=psutil.Process(h['pid']);assert p.create_time()==h['create_time'] and p.cmdline()==h['command']
    status=s.read(ROOT/c['state_root']/'families'/h['family']/'status.json')
    assert status['active_pid'] is None
    error=ROOT/c['state_root']/'launches'/h['family']/'stderr.log';assert not error.read_bytes()
    handles.append(dict(actual_handle=h,status=status))
s.write(folder/'actual7live_controllers_not577_completed_models.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    handles=handles,CPU_settings=c['CPU_settings'],CPU_resource_lock=c['CPU_resource_lock'],
    GPU_settings=c['GPU_settings'],GPU_resource_lock=c['GPU_resource_lock'],new577_completed_models_claimed=0,
    original_GPU_source_phase_status=s.read(Path(c['artifact_root'])/'neural/source_controller/gpu_status.json'),
    all_original_experiments_complete=False))
tc=tab.read(ROOT/tab.CONFIG);rows=[];closures=[]
for seed in [1103,1104]:
    e=next(e for e in tc['entries'] if e['case']['family']=='tranad64' and e['case']['dataset']=='smap' and e['case']['seed']==seed)
    b=tab.ready_case(tc,e);assert b is not None and len(b['dataset']['entities'])==54
    review=tab.read(b['lineage']['review_proof']);native=tab.read(b['lineage']['native_proof'])
    module=tab.component(e);nc=tab.read(ROOT/module.CONFIG)
    for phase in ['native','review']:
        attempt,out=module.paths(nc,e['case'],phase);guard=tab.read(attempt/'guard_receipt.json')
        assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted']
        if seed==1104:
            copy(out/module.REPORTS[phase],f'actual1104_{phase}_proof.json');copy(attempt/'guard_receipt.json',f'actual1104_{phase}_guard.json')
    if seed==1104:
        copy(b['lineage']['source_binding']['result_path'],'actual1104_source_result.json')
        copy(e['case']['source_guard'],'actual1104_source_guard.json')
    for policy,view in review['all_label_view_results'].items():
        for ratio,metric in view['metrics']['strict'].items():
            p=metric['micro'];ci=metric['f1_ci95'];diag=view['primary_protocol_diagnostics']
            rows.append(dict(method='TranAD_reference_float64',dataset='SMAP_full54',protocol='original_full109_v2',
                seed=seed,label_policy=policy,validation_false_alarm_percent=ratio,threshold=metric['calibration']['threshold'],
                precision=p['precision'],recall=p['recall'],F1=p['f1'],ROC_AUC=p['auroc'],AP=p['average_precision'],
                TP=p['true_positives'],FP=p['false_positives'],FN=p['false_negatives'],
                conditional_entity_F1_CI_low=ci['low'],conditional_entity_F1_CI_high=ci['high'],
                PA_F1_diagnostic_same_threshold=diag['same_validation_threshold_entitywise_PA_diagnostic']['f1'] if ratio=='1' else '',
                five_training_seed_uncertainty_complete=False,common105_result=False))
    closures.append(dict(seed=seed,case=e['case'],lineage=b['lineage'],all54entities_verified=True,
        all_source_and_native_arrays_byte_equal=native['all_array_dtype_shape_content_bytes_identical'],
        actual_all4label_fresh_review_closed=True,common105_result=False))
assert len(rows)==24
stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
(folder/'actual2seeds_all4labels_all3VAL_thresholds_not5seed_CI.csv').write_text(stream.getvalue(),encoding='utf-8',newline='')
s.write(folder/'actual2_of5_original_SMAP_seed_closures_not_common105.json',dict(records=closures,
    original_all15_native_cases_remain=True,five_training_seed_uncertainty_complete=False,all_original_experiments_complete=False))
incident=ROOT/'experiments/runs/fm_TAB_full878_original_metric_suite_v1/attempts/5eedb87987df7781d41c/e006'
g=tab.read(incident/'guard_receipt.json');assert g['resource_guard_aborted'] and g['exit_code']!=0 and not g['passed']
assert g['abort_snapshot']['commit_available_bytes']<tc['settings']['emergency_commit_headroom_bytes']
copy(incident/'guard_receipt.json','actual_TAB_A7_resource_abort_guard.json')
copy(Path(tc['artifact_root'])/'cases/5eedb87987df7781d41c/e006/progress.json','actual_TAB_A7_last_progress_not_completed.json')
s.write(folder/'TAB_resource_abort_is_not_algorithm_failure_or_completion.json',dict(
    original_guard=dict(settings=tc['settings'],resource_lock=tc['resource_lock']),
    completed_preceding_entities=6,incomplete_entity='SMAP_A-7',incomplete_case_key='5eedb87987df7781d41c',
    retry_performed=False,resource_thresholds_lowered=False,algorithm_accuracy_failure_claimed=False,
    remaining_resource_aborted_metric_work_complete=False))
copy(__file__,'publication_source.py')
s.write(folder/'archive_validation.json',dict(tests_passed=28,new_matched_source_jobs_registered=577,
    matched_case_coverage=873,matched_label_views=2037,actual_original_SMAP_seed_closures=2,
    new577_completed_models_claimed=0,original_research_scope=c['original_scope'],all_original_experiments_complete=False,
    files_sha256={p.relative_to(folder).as_posix():s.sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}))
ara=ROOT/'projects/flow_matching_research/ara/execution/common105_matched577_and_second_TranAD_seed.2026-10-16_v1.json'
s.write(ara,dict(evidence=folder.relative_to(ROOT).as_posix(),new_matched_source_jobs=577,comparison_cases=873,label_views=2037,
    tests_passed=28,actual_original_SMAP_seed_closures=2,new577_actual_completed_models_claimed=0,
    TAB_A7_resource_aborted_gap_preserved=True,original_scope=c['original_scope'],all_original_experiments_complete=False))
paths+=[ara.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()]
git=lambda root,*a:subprocess.check_output(['git','-C',str(root),*a],text=True).strip()
assert not git(ROOT,'diff','--cached','--name-only')
s.write(ROOT/'tmp/common105_matched577_publication_manifest_20261016_v1.json',dict(
    previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths),tests_passed=28,new_matched_source_jobs=577,actual_original_SMAP_seed_closures=2,live_family_controllers=7)))
