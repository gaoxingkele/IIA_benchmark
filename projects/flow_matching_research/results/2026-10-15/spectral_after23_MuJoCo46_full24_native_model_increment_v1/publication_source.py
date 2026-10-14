"""Publish complete five-trained-model MuJoCo statistics and the full24 catalog."""
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

root=Path(__file__).resolve().parents[1]
base='projects/flow_matching_research/results/2026-10-15/spectral_after23_MuJoCo46_full24_native_model_increment_v1'
reviewbase='projects/flow_matching_research/results/2026-10-15/spectral_mujoco_seed46_full_native_review_v1'
def read(path):return json.loads((root/path).read_text())
catalog=read(base+'/unique24_full_native_model_catalog.json')
assert catalog['full_native_generators']==len(catalog['records'])==24
assert catalog['reused_previous_full_native_reviews']==23 and catalog['previous23_closed_records_unchanged']
assert catalog['original_registered_generation_jobs']==70
assert catalog['original_total_optimizer_updates']==395000 and catalog['original_total_training_minibatches']==790000
assert catalog['original_total_full_evaluator_runs']==480
assert not catalog['full_generative_reinference_certified'] and not catalog['learned_evaluator_retraining_certified']
cohorts=read(base+'/five_cohort_completion_and_per_model_metrics.json')['cohorts']
cohort=next(c for c in cohorts if c['dataset']=='mujoco')
assert cohort['independently_reviewed_training_seeds']==[42,43,44,45,46] and cohort['full_five_seed_cohort_complete']
assert not cohort['missing_original_training_seeds']
assert all(v['independent_training_models']==5 and v['evaluator_runs']==25 and v['degrees_of_freedom']==4
           and v['full_cohort_mean'] is not None and v['between_training_model_sample_STD'] is not None
           and len(v['between_training_model_CI95'])==2 for v in cohort['metrics'].values())
fmri=next(c for c in cohorts if c['dataset']=='fmri')
assert not fmri['full_five_seed_cohort_complete'] and fmri['missing_original_training_seeds']==[43]
assert all(v['full_cohort_mean'] is None for v in fmri['metrics'].values())
xml=root/'tmp/Spectral_MuJoCo46_full24_five_model_validation_20261015_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==31
assert all(int(s.get('failures',0))==int(s.get('errors',0))==0 for s in suites)
folder=root/base
shutil.copyfile(xml,folder/'validation.xml')
(folder/'complete_MuJoCo_five_training_model_statistics.json').write_text(json.dumps(cohort,indent=2)+'\n')
(folder/'publication_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n')
lines=['','Complete five-trained-model MuJoCo cohort (seeds42..46):','',
    '| Metric | Five-model mean | Between-model sample STD | 95% t CI, df4 |',
    '|---|---:|---:|---|']
for metric,v in cohort['metrics'].items():
    lo,hi=v['between_training_model_CI95']
    lines.append(f"|{metric}|{v['full_cohort_mean']:.10g}|{v['between_training_model_sample_STD']:.10g}|[{lo:.10g},{hi:.10g}]|")
lines += ['','These are5training models, with5evaluator repeats nested inside each. '
    'Intervals describe variation in the5model-specific evaluator means on the fixed native data. '
    'The saved25evaluator measurements per metric were not counted as25training seeds. '
    '31relevant checks passed; independent arithmetic also rechecked every mean/STD/t interval. '
    'This closure did not regenerate samples or independently retrain learned evaluators. '
    'fMRI43 and all70generation jobs/45-paper/681-method/273-data obligations remain.']
with (folder/'README.md').open('a',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
targets=[base,reviewbase,
    'configs/reproducibility/fm_spectral_after23_MuJoCo46_full24_native_increment.2026-10-15_v1.json',
    'configs/reproducibility/fm_spectral_mujoco_seed46_full_native_review.2026-10-15_v1.json',
    'configs/reproducibility/fm_spectral_mujoco46_review_20261015_v1']
paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    raw=subprocess.check_output(['git',*args,'--',*targets],cwd=root)
    paths.update(p for p in raw.decode().splitlines() if not p.endswith('.log'))
assert paths and not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root).strip()
manifest=dict(previous_root='340369c244918d169948e909b73d722189444683',
    previous_parent='0bd2175e120679f399e7f3d6e12f10db4923e3ca',paths=sorted(paths))
(root/'tmp/spectral_MuJoCo46_full24_five_model_publication_manifest_20261015_v1.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(exact_files=len(paths),full_artifact_reviewed_models=24,required_original_generation_jobs=70,
    MuJoCo_complete_original_model_seeds=[42,43,44,45,46],full_five_model_mean_STD_CI95_complete=True,
    validation_tests_passed=31,all_original_experiments_complete=False)))
