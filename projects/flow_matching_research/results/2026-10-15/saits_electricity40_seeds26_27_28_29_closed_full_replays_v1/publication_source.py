"""Publish the fourth fully reviewed model in the original five-seed cohort."""
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[1]
base = 'projects/flow_matching_research/results/2026-10-15/saits_electricity40_seeds26_27_28_29_closed_full_replays_v1'
folder = root / base
closure = json.loads((folder / 'four_complete_original_models_and_full_grid_reviews.json').read_text(encoding='utf-8'))
assert closure['all_four_original_trainings_full_native_models_and_independent_reviews_passed']
assert closure['fully_verified_original_seeds'] == [26, 27, 28, 29]
assert closure['required_original_training_seeds'] == [26, 27, 28, 29, 30]
assert closure['reviewed_SAITS_BRITS_models'] == 43 and closure['required_SAITS_BRITS_models'] == 210
assert len(closure['per_model']) == 4 and closure['full_test_shape'] == [290, 100, 370]
assert not closure['complete_five_seed_aggregate'] and not closure['all_original_experiments_complete']
assert not closure['strict_TAB_result'] and not closure['author_equivalence_certified']
xml = root / 'tmp/SAITS40_seeds26_27_28_29_full43_validation_20261015_v1.xml'
suites = list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests', 0)) for s in suites) == 40
assert all(int(s.get('failures', 0)) == int(s.get('errors', 0)) == 0 for s in suites)
shutil.copyfile(xml, folder / 'validation.xml')
(folder / 'publication_source.py').write_bytes(Path(__file__).read_bytes())
identity = 'saits_paper_saits_electricity_author_missing0.4_seed29'
config_root = 'configs/reproducibility/fm_saits_completion_replay_queue_v1'
configs = [p.relative_to(root).as_posix() for p in (root / config_root).glob(identity + '*.json')]
assert len(configs) == 2
auto = 'projects/flow_matching_research/results/2026-10-11/saits_automatic_complete_model_reviews_v1/' + identity
targets = [base, auto, *configs]
paths = set()
for args in [('diff', '--name-only'), ('ls-files', '--others', '--exclude-standard')]:
    raw = subprocess.check_output(['git', *args, '--', *targets], cwd=root)
    paths.update(p for p in raw.decode().splitlines() if not p.endswith('.log'))
assert paths and not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=root).strip()
manifest = dict(previous_root='3e71bae5c3d3d18b485ec592eee575146cc14740',
                previous_parent='a2fdd2aac7eae7807d33246c34d20597ee460820', paths=sorted(paths))
(root / 'tmp/SAITS40_seed29_full43_publication_manifest_20261015_v1.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(exact_files=len(paths), complete_original_models=43, required_original_models=210,
                     original_SAITS_Electricity40_seeds_closed=[26, 27, 28, 29], validation_tests_passed=40)))
