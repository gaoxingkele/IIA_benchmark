"""Publish two completed local verification cases with all remaining scope intact."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from scripts.flow_matching.prepare_spectral_original_data import sha

root=Path(__file__).resolve().parents[1]
timebase='projects/flow_matching_research/results/2026-10-14/timesnet_SMAP1103_complete_two_stage_replay_v1'
spectralbase='projects/flow_matching_research/results/2026-10-14/spectral_after22_MuJoCo45_full23_native_model_increment_v1'
reviewbase='projects/flow_matching_research/results/2026-10-14/spectral_mujoco_seed45_full_native_review_v1'
def read(path):return json.loads((root/path).read_text())
timepath=timebase+'/complete_two_stage_native_and_independent_saved_score_closure.json'
catalogpath=spectralbase+'/unique23_full_native_model_catalog.json'
time=read(timepath);catalog=read(catalogpath)
assert time['full_registered_two_stage_audit_complete'] and time['separately_registered_point_audit_passed']
assert not time['separately_registered_point_audit_pending'] and time['original_and_replay_score_archives_same_sha256']
assert time['full_entities']==51 and time['full_test_points']==408160
assert catalog['full_native_generators']==len(catalog['records'])==23
assert catalog['reused_previous_full_native_reviews']==22 and catalog['previous22_closed_records_unchanged']
new=next(r for r in catalog['records'] if r['canonical_id']=='spectral_original__diffusion_ts__mujoco__released_author_data__seed45')
assert new['optimizer_updates']==14000 and new['training_minibatches']==28000
assert new['complete_reference_shape']==[10000,24,14] and new['complete_generated_shape']==[10005,24,14]
cohorts=read(spectralbase+'/five_cohort_completion_and_per_model_metrics.json')['cohorts']
cohort=next(c for c in cohorts if c['dataset']=='mujoco')
assert cohort['independently_reviewed_training_seeds']==[42,43,44,45] and not cohort['full_five_seed_cohort_complete']
assert all(v['full_cohort_mean'] is None and v['between_training_model_CI95'] is None for v in cohort['metrics'].values())
xml=root/'tmp/TimesNet_complete_Spectral_MuJoCo45_validation_20261014_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==68
assert all(int(s.get('failures',0))==int(s.get('errors',0))==0 for s in suites)
shutil.copyfile(xml,root/spectralbase/'validation.xml')
joint='projects/flow_matching_research/results/2026-10-14/timesnet_complete_and_MuJoCo45_full23_increment_v1'
folder=root/joint;folder.mkdir(parents=True,exist_ok=False)
summary=dict(captured_utc=datetime.now(timezone.utc).isoformat(),original_scope=time['original_scope'],
    TimesNet=dict(task='time_series_anomaly_detection',canonical_id=time['canonical_id'],
        complete_original_registered_two_stage_audit=True,bitwise_score_archive=True,parameters=28132633,
        full_entities=51,train_points=101565,validation_points=26684,test_points=408160,
        primary_metrics=time['independently_recomputed_primary_metrics'],
        conditional_entity_cluster_f1_ci95=time['conditional_entity_cluster_f1_ci95'],
        configured_epochs=3,full_local_training_windows=1044,configured_optimizer_steps=27,
        measured_per_epoch_update_trace_available=False,full_TAB_range_metrics_completed=False),
    Diffusion_TS=dict(task='unconditional_time_series_generation',canonical_id=new['canonical_id'],
        full_original_updates=14000,full_original_training_minibatches=28000,
        complete_reference_shape=[10000,24,14],complete_generated_shape=[10005,24,14],
        full_original_metrics_and_five_evaluator_repeats=new['metrics'],
        artifact_reviewed_generators=23,required_generators=70,MuJoCo_closed_training_seeds=[42,43,44,45],
        MuJoCo_required_training_seeds=[42,43,44,45,46],full_five_seed_summary_complete=False,
        checkpoint_tensors_restored_bitwise=True,full_saved_reference_and_generated_arrays_verified=True,
        independent_generative_reinference_performed=False,learned_evaluators_independently_retrained=False),
    validation_tests_passed=68,additional_optimizer_updates=0,additional_training_seed_slots=0,
    historical_author_equivalence_certified=False,strict_TAB_training_harness_certified=False,
    all_original_experiments_complete=False,
    source_receipts=[dict(path=p,sha256=sha(root/p)) for p in [timepath,catalogpath]])
(folder/'completed_TimesNet_audits_and_full23_generation_catalog.json').write_text(json.dumps(summary,indent=2)+'\n')
(folder/'publication_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n')
shutil.copyfile(xml,folder/'validation.xml')
(folder/'README.md').write_text(
    'Two verification cases are complete; they address different tasks and do not form '
    'a shared algorithm ranking.68 relevant checks passed.\n\n'
    'TimesNet SMAP seed1103: both the complete native checkpoint inference and standalone '
    'saved-score/raw-data/point/core/event/bootstrap audit passed. All51entities/408160test '
    'points and the entire train/validation/test score archive match bitwise. '
    'Validation1% noPA F1=0.0453082580,AUROC=0.4362539773,AP=0.1213720771. '
    'Original local training uses1044nonoverlapping/end-aligned windows,3epochs and '
    '27configured updates; the actual update trace was not saved. This does not certify '
    'author-harness or fullTAB-training-harness equivalence.\n\n'
    'Diffusion-TS MuJoCo seed45: original14000updates/28000minibatches, full10000x24x14 '
    'reference and10005x24x14 generated arrays, full checkpoint tensor restoration and '
    'four5-repeat evaluator statistics passed. MeanFID=0.0156868563,correlational=0.2093448890, '
    'discriminative=0.017,predictive=0.0075126674. This audit did not independently '
    'regenerate samples or retrain learned evaluators. Generation artifact-review '
    'catalog now23/70; MuJoCo42..45=4/5, so full five-seed mean/STD/CI remain null '
    'until seed46. Earlier22catalog records are unchanged.\n\n'
    'All original35baseline slots/291data controls,five generation seeds/all70jobs '
    'and45-paper/681-method/273-data scope remain. Full range evaluations, remaining '
    'models/seeds and historical fidelity gaps remain separate obligations.\n')
for pending in [
    'projects/flow_matching_research/results/2026-10-14/timesnet_SMAP1103_constant_fit_full_replay_v1/README.md',
    'projects/flow_matching_research/results/2026-10-14/timesnet_SMAP1103_bitwise_native_complete_point_pending_v1/README.md']:
    with (root/pending).open('a',encoding='utf-8',newline='\n') as stream:
        stream.write('\nFinal stage completion update: the independently registered standalone '
            'point audit has now passed; both original stage obligations are complete. '
            'See ../timesnet_SMAP1103_complete_two_stage_replay_v1/README.md. '
            'Earlier pending snapshots are preserved.\n')
targets=[timebase,spectralbase,reviewbase,joint,
    'configs/reproducibility/fm_spectral_after22_MuJoCo45_full23_native_increment.2026-10-14_v1.json',
    'configs/reproducibility/fm_spectral_mujoco_seed45_full_native_review.2026-10-14_v1.json',
    'configs/reproducibility/fm_spectral_mujoco45_review_20261014_v1',
    'projects/flow_matching_research/results/2026-10-14/timesnet_SMAP1103_constant_fit_full_replay_v1/point',
    'projects/flow_matching_research/results/2026-10-14/timesnet_SMAP1103_constant_fit_full_replay_v1/README.md',
    'projects/flow_matching_research/results/2026-10-14/timesnet_SMAP1103_bitwise_native_complete_point_pending_v1/README.md']
paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    raw=subprocess.check_output(['git',*args,'--',*targets],cwd=root)
    paths.update(p for p in raw.decode().splitlines() if not p.endswith('.log'))
assert paths and not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root).strip()
manifest=dict(previous_root='46fa8de1cc9ebf0e4ab430daac29714c4fefd7f4',
    previous_parent='ce7df283925fc8baac6c1ae9a94d4b775e19da8b',paths=sorted(paths))
(root/'tmp/TimesNet_complete_MuJoCo45_full23_publication_manifest_20261014_v1.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(exact_files=len(paths),TimesNet_two_stage_complete=True,generation_artifact_reviewed_models=23,
    full_MuJoCo_cohort_complete=False,validation_tests_passed=68)))
