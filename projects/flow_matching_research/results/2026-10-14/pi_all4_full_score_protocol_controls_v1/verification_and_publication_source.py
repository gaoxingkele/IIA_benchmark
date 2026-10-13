from datetime import datetime,timezone
import json,subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
import psutil
import numpy as np
from scripts.flow_matching import audit_pi_closed_score_protocol_controls_v1 as audit

root=audit.ROOT
folder=root/'projects/flow_matching_research/results/2026-10-14/pi_all4_full_score_protocol_controls_v1'
cfgpath='configs/reproducibility/fm_pi_all4_closed_protocol_controls.2026-10-14_v1.json'
cfg=audit.review.read(cfgpath)
report=audit.review.read(cfg['output_report'])
assert len(report['records'])==4 and len(report['branch_seed_protocol_summaries'])==8
assert all(len(r['controls'])==4 for r in report['records'])
checks=[]
for row in report['records']:
    controls=row['controls']
    def select(calibration,PA):return next(c for c in controls if c['calibration']==calibration and c['label_dependent_PA']==PA)
    pooled=select('author_pooled_train_test',False);training=select('train_only_same_percentile',False)
    PA=select('author_pooled_train_test',True)
    assert pooled['confusion']==training['confusion']
    item=next(c for c in cfg['closed_cases'] if c['canonical_id']==row['canonical_id'])
    closed=audit.review.read(item['independent_proof'])
    reviewcfg=audit.review.read(closed['configuration'])
    nativecfg=audit.review.read(reviewcfg['completed_replay_configuration'])
    case,job,result=audit.review.binding(nativecfg)
    scorepath=root/case['job']['output_directory']/'checkpoints/author_scores.npz'
    assert audit.review.digest(scorepath)==row['source_score_sha256']
    with np.load(scorepath,allow_pickle=False) as scores:
        assert np.array_equal((scores['test_energy']>training['threshold']).astype(int),scores['raw_predictions'])
    assert row['PA_added_fraction_of_reported_true_positives']>.98
    checks.append(dict(canonical_id=row['canonical_id'],seed=row['seed'],branch=row['branch'],
                       F1_no_PA=pooled['metrics']['f1'],F1_original_PA=PA['metrics']['f1'],
                       pooled_threshold=pooled['threshold'],train_only_threshold=training['threshold'],
                       identical_prediction_counts_at_train_only_threshold=True,
                       all87800_prediction_bits_identical_at_train_only_threshold=True,
                       original_AUROC=row['ranking_metrics']['auroc'],original_AP=row['ranking_metrics']['average_precision'],
                       PA_added_true_positive_points=row['PA_added_true_positive_points'],
                       PA_added_fraction_of_reported_true_positives=row['PA_added_fraction_of_reported_true_positives'],
                       dropped_original_test_tail_points=row['original_dropped_test_tail_points']))
for item in cfg['closed_cases']:
    proof=audit.review.read(item['independent_proof']);reviewcfg=audit.review.read(proof['configuration'])
    nativepath=reviewcfg['completed_replay_configuration'];nativecfg=audit.review.read(nativepath)
    branch='causal' if proof['branch']=='causal_axis_repaired' else 'source'
    target=folder/'closed_full_model_evidence'/(branch+str(proof['seed']))
    target.mkdir(parents=True,exist_ok=True)
    sources=[(item['independent_proof'],'independent_full_array_proof.json'),
             (proof['configuration'],'independent_configuration.json'),
             (reviewcfg['state_root']+'/receipt.json','independent_guard_receipt.json'),
             (nativepath,'native_configuration.json'),
             (reviewcfg['completed_replay_report'],'native_full_model_report.json'),
             (nativecfg['state_root']+'/receipt.json','native_guard_receipt.json'),
             (nativecfg['independent_completion'],'original_full_budget_point_audit.json')]
    for source,name in sources:
        data=(root/source).read_bytes()
        if (target/name).exists():
            assert (target/name).read_bytes()==data
        else:
            (target/name).write_bytes(data)
suites=list(ET.parse(folder/'validation.xml').getroot().iter('testsuite'))
assert all(int(s.get('failures',0))==int(s.get('errors',0))==0 for s in suites)
assert sum(int(s.get('tests',0)) for s in suites)==44
live=[]
for name,pid,ctime in [('Pi_full1140_source',37476,1791770405.0394766),
                       ('TCCM_original_full5_training_recovery',37976,1791741883.5107028),
                       ('SAITS_original_full210',2972,1791918817.7624688),
                       ('CFM_original_full285',40444,1791922321.0307884)]:
    p=psutil.Process(pid);assert p.create_time()==ctime
    live.append(dict(name=name,pid=p.pid,create_time=p.create_time(),command=p.cmdline(),
                     actual_children=[dict(pid=c.pid,create_time=c.create_time(),command=c.cmdline()) for c in p.children(recursive=True)]))
registry=audit.review.read('configs/models/fm_comparator_pi_transformer.json')
evidence=dict(captured_utc=datetime.now(timezone.utc).isoformat(),original_scope=cfg['original_scope'],
               original_Pi_jobs_preserved=1140,original_Pi_training_seeds_preserved=[42,1103,1104,1105,1106,1107],
               actual_closed_full_models=4,actual_fixed_score_calibration_PA_controls=16,
               each_original_full_budget=dict(epochs=3,optimizer_minibatches=1554,parameters=1930289,train_score_coordinates=13238200,test_score_coordinates=87800),
               individually_checked_original_controls=checks,
               high_local_F1_same_scores_changes_at_label_dependent_PA=True,
               train_only_posthoc_calibration_changes_predictions_in_these_cases=False,
               observation_does_not_exonerate_test_selected_early_stopping=True,
               source_mirror_citation=registry['citation'],source_mirror_commit=registry['source_commit'],
               preserved_author_fidelity_boundaries=registry['remaining_gaps'],
               actual_live_remaining_original_queue_handles=live,validation_tests_passed=44,
               extra_model_inference=0,extra_optimizer_updates=0,scientific_worker_stopped=False,
               complete_six_seed_cohort_or_between_seed_uncertainty_proved=False,
               strict_TAB_result=False,author_equivalence_certified=False,all_original_experiments_complete=False)
write=lambda p,v:p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
write(folder/'verified_high_F1_protocol_effect_and_actual_scope.json',evidence)
(folder/'verification_and_publication_source.py').write_bytes(Path(__file__).read_bytes())
paths=['scripts/flow_matching/audit_pi_closed_score_protocol_controls_v1.py','tests/test_pi_closed_score_protocol_controls_v1.py',cfgpath]
paths += [p.relative_to(root).as_posix() for p in folder.rglob('*') if p.is_file()]
head=lambda cwd:subprocess.check_output(['git','-C',str(cwd),'rev-parse','HEAD'],text=True).strip()
write(root/'tmp/pi_all4_protocol_controls_publication_manifest_20261014.json',dict(previous_root=head(root),
      previous_parent=head(root.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(files=len(paths),tests=44,closed_models=4,calibration_PA_controls=16,
                     source_F1_min=min(c['F1_no_PA'] for c in checks),source_F1_max=max(c['F1_original_PA'] for c in checks))))
