"""Publish actual full model closure, full negative data audit and live next work."""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import psutil

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha
from scripts.flow_matching import audit_otcfm_multiome_full_reference_overlap_v1 as audit
from scripts.flow_matching import run_spectral_full70_artifact_reviews_v1 as watcher
from scripts.flow_matching import prepare_otcfm_multiome13176_public_independent_v1 as preprocessing

BASE = 'projects/flow_matching_research/results/2026-10-16/GRASP45_full_reference_overlap_and_automatic70_v1'
GRASP = 'projects/flow_matching_research/results/2026-10-16/grasp_tau1_seed1106_full45_model_increment_v1'
folder = ROOT / BASE; folder.mkdir(parents=True, exist_ok=False)


def save(name, value):
    with (folder / name).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')


def validation(path, expected):
    suites = list(ET.parse(path).getroot().iter('testsuite'))
    assert sum(int(x.get('tests', 0)) for x in suites) == expected
    assert all(int(x.get('failures', 0)) == int(x.get('errors', 0)) == 0 for x in suites)


validation(ROOT / GRASP / 'validation.xml', 34)
for src, name, count in [
    ('F:/aicoding/IIA_Data/experiments/spectral_full70_artifact_reviews_v1_tests.xml', 'automatic70_validation48.xml', 48),
    ('F:/aicoding/IIA_Data/experiments/otcfm_multiome13176_public_independent_v2_tests.xml', 'reference_and_public_input_validation41.xml', 41)]:
    validation(src, count); shutil.copyfile(src, folder / name)
failed = Path('F:/aicoding/IIA_Data/experiments/otcfm_multiome13176_public_independent_v1_tests.xml')
assert sum(int(x.get('failures', 0)) for x in ET.parse(failed).getroot().iter('testsuite')) == 1
save('preserved_pre_registration_test_failure.json', dict(original_path=str(failed), original_bytes_sha256=sha(failed),
    exact_xml_utf8_content=failed.read_bytes().decode('utf-8'),
    diagnosed_issue='A double-precision dense Gram reference was compared to intentional float32 author-loader PC output at an inappropriate1e-8 threshold.',
    correction='Quantize the independent dense PC reference to the actual registered float32 representation, then compare Gram products in float64 at1e-12. Production PCA, output dtype, projection tolerances and all resource policies remain unchanged.'))

c = read(ROOT / audit.CONFIG); audit.verify(c)
out = ROOT / c['output_directory']; state = ROOT / c['state_root']
guard = read(state / 'guard_receipt.json'); report = read(out / 'full_reference_overlap_report.json')
preprocessing.validate_alignment_evidence(report, guard)
assert guard['report_sha256'] == sha(out / 'full_reference_overlap_report.json')
assert report['configuration_sha256'] == sha(ROOT / audit.CONFIG)
with np.load(out / 'all_author_reference_mappings.npz', allow_pickle=False) as arrays, \
        np.load(out / 'all_reference_coordinate_comparisons.npz', allow_pickle=False) as coordinates:
    primary = arrays['primary']; mapped = arrays['diagnostic_mapping']; days = arrays['author_days']
    assert primary.shape == mapped.shape == days.shape == (33701,)
    assert np.array_equal(primary, mapped)
    good = primary >= 0; missing = np.flatnonzero(~good)
    assert np.count_nonzero(good) == 33697 and missing.tolist() == [1755, 3095, 3424, 4330]
    assert len(np.unique(primary[good])) == 33697
    assert np.array_equal(arrays['public_days'][primary[good]], days[good])
    assert set(arrays['public_donors'][primary[good]].tolist()) == {'32606'}
    assert not coordinates['mismatch_counts'].any()
    assert np.array_equal(coordinates['full_expression_match'], good)
    assert np.array_equal(coordinates['full_reference_donor_match'], good)
    assert np.isfinite(coordinates['maximum_absolute_errors']).all()
    maximum_error = float(coordinates['maximum_absolute_errors'][good].max())
    selected = np.flatnonzero((arrays['public_donors'] == '32606') & np.isin(arrays['public_days'], c['days']))
    public_only = np.setdiff1d(selected, primary[good])
    assert len(public_only) == 7
journal = [json.loads(line) for line in (out / 'all_author_anchor_outcomes.jsonl').read_text().splitlines()]
assert [r['author_row'] for r in journal] == list(range(33701))
assert all(not r['reference_candidates'] and not r['global_same_day_candidates'] for r in journal if r['author_row'] in missing)
with h5py.File(ROOT / c['inputs']['author']['path'], 'r') as af, h5py.File(ROOT / c['inputs']['public']['path'], 'r') as pf:
    author_ids = audit.source.column(af['obs'], af['obs'].attrs['_index'])
    public_ids = audit.source.column(pf['obs'], pf['obs'].attrs['_index'])
    public_days = audit.source.column(pf['obs'], 'day').astype(int)
    save('unresolved_reference_and_public_only_cells.json', dict(
        unresolved_author_cells=[dict(author_row=int(i), author_cell_id=str(author_ids[i]), day=int(days[i])) for i in missing],
        public_only_reference_candidates=[dict(public_row=int(i), public_cell_id=str(public_ids[i]), day=int(public_days[i])) for i in public_only],
        provenance_difference_cause_known=False, biological_identity_proven=False,
        cell_removal_or_imputation_performed=False))
for src, name in [(out / 'full_reference_overlap_report.json', 'full_reference_overlap_report.json'),
    (state / 'guard_receipt.json', 'full_reference_actual_guard.json'), (state / 'launch.json', 'full_reference_launch.json')]:
    shutil.copyfile(src, folder / name)
artifacts = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]
save('full_reference_independent_artifact_checks.json', dict(passed=True, all33701_journal_rows_present_once=True,
    all_mapped_rows_have_full23418_gene_and_implicit_zero_comparisons=True, matched_rows=33697,
    full_coordinate_comparisons=789116346, maximum_absolute_error_for_matched_rows=maximum_error,
    unmatched_rows_have_no_assumed_zero_error_or_identity=True, public_only_candidate_rows=7,
    strict_full_reference_identity_passed=False, paper_PCA_input_produced_by_audit=False,
    artifacts=artifacts, actual_worker_environment=dict(python=sys.version,
        packages={k: importlib.metadata.version(k) for k in ['numpy', 'scipy', 'h5py', 'scikit-learn']})))

wc = read(ROOT / watcher.CONFIG); cases, catalog = watcher.verify(wc)
assert len(cases) == 70 and len(catalog['records']) == 30
pc = read(ROOT / preprocessing.CONFIG); preprocessing.verify(pc)
handles = {}
for label, module, cfg in [('automatic70', watcher.MODULE, wc), ('independent_public_input', preprocessing.MODULE, pc)]:
    base = ROOT / cfg['state_root']; launch = read(base / 'launch.json'); p = psutil.Process(launch['pid'])
    assert p.create_time() == launch['create_time'] and p.cmdline() == launch['command'] and module in p.cmdline()
    handles[label] = dict(pid=p.pid, create_time=p.create_time(), command=p.cmdline(),
        children=[dict(pid=x.pid, create_time=x.create_time(), command=x.cmdline()) for x in p.children(recursive=True)])
    shutil.copyfile(base / 'launch.json', folder / (label + '_launch.json'))
    shutil.copyfile(base / 'status.json', folder / (label + '_status_at_publication.json'))
save('actual_next_work_handles.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(), handles=handles))
closure = read(ROOT / GRASP / 'actual_full45_model_increment_closure.json')
assert closure['actual_current_unique_complete_models'] == 45
assert closure['complete_Tau1_Swan_seeds'] == [1103, 1104, 1105, 1106]
assert not closure['full_ten_seed_Tau1_cohort_complete'] and len(closure['recipes']) == 1
save('iteration_validation.json', dict(passed=True, relevant_tests_passed=123, complete_GRASP_models=45,
    registered_GRASP_models=9120, original_cohorts=480, original_training_seeds=10,
    new_GRASP_model=closure['canonical_id'], new_model_native_and_independent_review_complete=True,
    full_reference_diagnostic_complete=True, strict_full_reference_identity_passed=False,
    automatic_generation_artifact_review_scope=70, preserved_full_generation_reviews=30,
    independent_public_paper_cohort_registered=True, independent_public_paper_cohort_completed=False,
    original_scope=pc['original_scope'], all_original_experiments_complete=False))
shutil.copyfile(__file__, folder / 'publication_source.py')
text = '''GRASP Tau1/SWaN/swan_sf seed1106 completed full1500epochs,93000optimizer updates, full3935validation and72000test points, all one original5x10 scoring recipe, guarded native checkpoint reinference and independent complete-array/metric review. The canonical completed ledger is45/9120. Validation-only/noPA F1=0.5253541754274031; retrospective test-label-optimized F1=0.5519139943175413, AP=0.5507189776934553, ROC=0.7411391587342671. Tau1 SWaN has4/10 original training seeds; no full ten-seed interval or author/TAB equivalence is claimed. The neighboring grasp_tau1_seed1106_full45_model_increment_v1 packet contains complete source/guard/native/independent evidence.

The full OT-CFM Multiome reference audit completed with an actual successful guard. All33701 author rows were examined.33697 rows matched donor32606 and all23418 gene coordinates, including implicit zeros:789116346 coordinate comparisons, no mismatching mapped rows. Four author rows1755,3095,3424,4330 (all day2) had no expression-anchor candidate even in the global same-day search. Seven public reference candidates have no matched author row. This is a measured discrepancy; its provenance cause and biological identity are not proved. The original strict whole-reference gate remains failed. No PCA/training input was generated by this diagnostic.

A separate independent public paper-described donor13176 track is registered and live:all35384 cells/all23418 genes/days2,3,4,7, fixed encoding, fresh first5 centered PCs and exact released-author whitening. The failed historical identity gate remains explicit; historical paper target roster/PCA artifact identity are unknown. No target row is filtered and no wrong donor is substituted. Completion requires its actual successful guard and preprocessing report; registration alone is not completion.

The full70-generator artifact-review watcher is live, preserving30 already closed reviews and watching all original14cohorts/five-seed obligations, including the explicit full-budget fMRI43 recovery. It uses the unchanged native tensor/array/four-evaluator audit, existing shared lock and resource safeguards. Artifact review does not rerun generation or learned evaluators.123 relevant checks passed; the failed pre-registration float32-reference test is preserved with its diagnosis. The full45-paper/681-method-baseline-ablation/273-data-task/702-target objective remains active.
'''
(folder / 'README.md').write_text(text, encoding='utf-8', newline='\n')
for path in [folder, ROOT / GRASP]:
    (path / '.gitattributes').write_text('*.py -text\n*.json -text\n', encoding='utf-8', newline='\n')
identity = closure['canonical_id']
targets = [BASE, GRASP,
    'projects/flow_matching_research/results/2026-10-12/grasp_full9120_completion_replay_v1/completed_models/' + identity,
    *[p.relative_to(ROOT).as_posix() for p in (ROOT / 'configs/reproducibility/fm_grasp_completion_replay_children_v1').glob(identity + '*.json')],
    audit.MODULE.replace('.', '/') + '.py', audit.TEST, audit.CONFIG,
    watcher.MODULE.replace('.', '/') + '.py', watcher.TEST, watcher.CONFIG,
    preprocessing.MODULE.replace('.', '/') + '.py', preprocessing.TEST, preprocessing.CONFIG]
paths = set()
for args in [('diff', '--name-only'), ('ls-files', '--others', '--exclude-standard')]:
    paths.update(p for p in subprocess.check_output(['git', *args, '--', *targets], cwd=ROOT, text=True).splitlines() if not p.endswith('.log'))
assert paths and not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT).strip()
manifest = dict(previous_root=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    previous_parent=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT.parent / 'flow-matching-research', text=True).strip(), paths=sorted(paths))
(ROOT / 'tmp/GRASP45_overlap_and_automatic70_exact_publication_20261016_v1.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(exact_files=len(paths), GRASP_closed=45, full_reference_mapped=33697,
    unresolved_author_rows=4, future_generator_audit_scope=70, relevant_tests_passed=123)))
