"""Publish one independently artifact-reviewed original generator model."""
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[1]
review = 'projects/flow_matching_research/results/2026-10-15/spectral_sines_seed45_released_track_full_native_review_v1'
base = 'projects/flow_matching_research/results/2026-10-15/spectral_after27_Sines45_full28_native_model_increment_v1'
folder = root / base
catalog = json.loads((folder / 'unique28_full_native_model_catalog.json').read_text(encoding='utf-8'))
assert catalog['full_native_generators'] == len(catalog['records']) == 28
assert catalog['previous27_closed_records_unchanged']
assert catalog['original_registered_generation_jobs'] == 70
assert catalog['original_total_optimizer_updates'] == 443000
assert catalog['original_total_training_minibatches'] == 886000
assert catalog['original_total_full_evaluator_runs'] == 560
assert not catalog['all_original_experiments_complete']
assert not catalog['full_generative_reinference_certified']
assert not catalog['learned_evaluator_retraining_certified']
assert not catalog['author_equivalence_certified'] and not catalog['strict_TAB_result']
cohorts = json.loads((folder / 'six_cohort_completion_and_per_model_metrics.json').read_text())['cohorts']
sines = next(c for c in cohorts if c['dataset'] == 'sines' and c['data_track'] == 'released_author_data')
assert sines['independently_reviewed_training_seeds'] == [42, 43, 44, 45]
assert sines['missing_original_training_seeds'] == [46]
assert not sines['full_five_seed_cohort_complete']
xml = root / review / 'validation_full36.xml'
suites = list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests', 0)) for s in suites) == 36
assert all(int(s.get('failures', 0)) == int(s.get('errors', 0)) == 0 for s in suites)
shutil.copyfile(xml, folder / 'validation.xml')
(folder / 'publication_source.py').write_bytes(Path(__file__).read_bytes())
for directory in [folder, root / review]:
    (directory / '.gitattributes').write_text('*.py -text\n*.json -text\n', encoding='utf-8')
targets = [base, review,
           'configs/reproducibility/fm_spectral_sines_seed45_released_track_full_native_review.2026-10-15_v1.json',
           'configs/reproducibility/fm_spectral_sines45_review_20261015_v1',
           'configs/reproducibility/fm_spectral_after27_Sines45_full28_native_increment.2026-10-15_v1.json']
paths = set()
for args in [('diff', '--name-only'), ('ls-files', '--others', '--exclude-standard')]:
    paths.update(p for p in subprocess.check_output(['git', *args, '--', *targets], cwd=root).decode().splitlines()
                 if not p.endswith('.log'))
assert paths and not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=root).strip()
manifest = dict(previous_root=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                previous_parent=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root.parent/'flow-matching-research',text=True).strip(), paths=sorted(paths))
(root / 'tmp/Spectral_Sines45_full28_publication_manifest_20261015_v1.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(exact_files=len(paths), complete_original_models=28, required_original_models=70,
                     Sines_released_track_training_seeds=[42, 43, 44, 45], validation_tests_passed=36)))
