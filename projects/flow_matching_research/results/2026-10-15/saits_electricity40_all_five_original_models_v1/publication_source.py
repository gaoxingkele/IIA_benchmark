"""Publish the fully verified original five-model imputation cohort."""
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[1]
base = 'projects/flow_matching_research/results/2026-10-15/saits_electricity40_all_five_original_models_v1'
folder = root / base
summary = json.loads((folder / 'full_five_seed_cohort_summary.json').read_text(encoding='utf-8'))
assert (summary['method'], summary['dataset'], summary['nominal_missing_ratio']) == ('saits', 'electricity', .4)
assert summary['full_five_seed_cohort_complete'] and summary['independently_trained_models'] == 5
assert summary['original_training_seeds'] == [26, 27, 28, 29, 30]
assert len(summary['per_original_seed']) == 5
assert summary['held_out_coordinates_per_model'] == 3537183
assert summary['full_test_grid_cells_per_model'] == 10730000
assert all(m['n'] == 5 and m['degrees_of_freedom'] == 4 for m in summary['summaries'].values())
assert not summary['all_original_experiments_complete']
assert not summary['author_equivalence_certified'] and not summary['strict_TAB_result']
assert not summary['original_prediction_bitwise_equivalence_claimed']
closure = json.loads((folder / 'closure_validation.json').read_text())
assert closure['passed'] and closure['verified_SAITS_BRITS_models'] >= 44
scalar = json.loads((folder / 'independent_actual_five_model_scalar_and_SciPy_t95_verification.json').read_text())
assert scalar['passed'] and scalar['independent_training_models'] == 5 and scalar['no_original_seed_excluded']
suites = list(ET.parse(folder / 'validation.xml').getroot().iter('testsuite'))
assert sum(int(s.get('tests', 0)) for s in suites) == 49
assert all(int(s.get('failures', 0)) == int(s.get('errors', 0)) == 0 for s in suites)
(folder / 'publication_source.py').write_bytes(Path(__file__).read_bytes())
(folder / '.gitattributes').write_text('*.py -text\n*.json -text\n', encoding='utf-8')
identity = 'saits_paper_saits_electricity_author_missing0.4_seed30'
configroot = 'configs/reproducibility/fm_saits_completion_replay_queue_v1'
configs = [p.relative_to(root).as_posix() for p in (root / configroot).glob(identity + '*.json')]
assert len(configs) == 2
auto = 'projects/flow_matching_research/results/2026-10-11/saits_automatic_complete_model_reviews_v1/' + identity
targets = [base, summary['frozen_configuration'], auto, *configs]
paths = set()
for args in [('diff', '--name-only'), ('ls-files', '--others', '--exclude-standard')]:
    paths.update(p for p in subprocess.check_output(['git', *args, '--', *targets], cwd=root).decode().splitlines()
                 if not p.endswith('.log'))
assert paths and not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=root).strip()
manifest = dict(previous_root='ccad164193b3ee11c2221fc4105663ba3519ecf3',
                previous_parent='f5005066786d003d8febbff968ad9edf4e3a0999', paths=sorted(paths))
(root / 'tmp/SAITS_electricity40_five_seed_publication_manifest_20261015_v1.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(exact_files=len(paths), verified_models=closure['verified_SAITS_BRITS_models'],
                     full_original_training_seeds=5, validation_tests_passed=49, summaries=summary['summaries'])))
