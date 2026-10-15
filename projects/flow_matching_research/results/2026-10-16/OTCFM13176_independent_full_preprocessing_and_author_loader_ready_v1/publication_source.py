"""Archive completed complete-cohort preprocessing and actual author-loader input."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from scripts.flow_matching import package_otcfm_independent_author_input_v1 as package

ROOT, read, sha = package.ROOT, package.read, package.sha
BASE = 'projects/flow_matching_research/results/2026-10-16/OTCFM13176_independent_full_preprocessing_and_author_loader_ready_v1'
folder = ROOT / BASE; folder.mkdir(parents=True, exist_ok=False)
config = read(ROOT / package.CONFIG); package.verify(config)
parent = read(ROOT / config['original_preprocessing'])
guard_path = ROOT / config['source_guard']; report_path = ROOT / config['source_report']
guard, report = read(guard_path), read(report_path)
assert guard['passed'] and guard['exit_code'] == 0 and not guard['resource_guard_aborted']
assert guard['report_sha256'] == sha(report_path) and report['configuration_sha256'] == sha(ROOT / config['original_preprocessing'])
for artifact in report['artifacts']:
    assert sha(ROOT / artifact['path']) == artifact['sha256']
safe_report_path = ROOT / config['output_directory'] / 'input_packaging_report.json'
safe = read(safe_report_path)
assert safe['passed'] and safe['all_arrays_pickle_free'] and safe['exact_original_author_loader_verified']
assert safe['configuration_sha256'] == sha(ROOT / package.CONFIG)
assert sha(ROOT / config['input_npz']) == safe['input_sha256']
with np.load(ROOT / config['input_npz'], allow_pickle=False) as source, \
        np.load(ROOT / config['source_input'], allow_pickle=False) as original, \
        np.load(ROOT / parent['output_directory'] / 'public13176_whitened_diagnostic.npz', allow_pickle=False) as diagnostic, \
        np.load(ROOT / parent['output_directory'] / 'public13176_PCA_and_whitening_parameters.npz', allow_pickle=False) as parameters:
    payload = {k: source[k] for k in source.files}; package.validate_payload(config, payload)
    for key in ['pcs', 'sample_labels', 'source_rows']:
        assert np.array_equal(source[key], original[key])
    independently = ((source['pcs'].astype(np.float64) - parameters['whitening_mean']) / parameters['whitening_scale']).astype(np.float32)
    assert np.allclose(independently, diagnostic['pcs'], rtol=2e-6, atol=2e-6)
    assert np.allclose(diagnostic['pcs'].mean(axis=0, dtype=np.float64), 0, atol=1e-6)
    assert np.allclose(diagnostic['pcs'].std(axis=0, dtype=np.float64), 1, atol=1e-6)
    arrays = {k: dict(dtype=str(v.dtype), shape=list(v.shape), finite=bool(np.isfinite(v).all()) if v.dtype.kind in 'fiu' else None)
              for k, v in payload.items()}
xml = Path('F:/aicoding/IIA_Data/experiments/otcfm_independent_author_input_package_v1_tests.xml')
suites = list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests', 0)) for s in suites) == 16
assert all(int(s.get('failures', 0)) == int(s.get('errors', 0)) == 0 for s in suites)
shutil.copyfile(xml, folder / 'validation16.xml')
for path, name in [(report_path, 'full_preprocessing_report.json'), (guard_path, 'actual_preprocessing_guard.json'),
    (safe_report_path, 'pickle_free_author_input_report.json')]:
    shutil.copyfile(path, folder / name)
summary = dict(passed=True, captured_utc=datetime.now(timezone.utc).isoformat(), data_track=config['data_track'],
    complete_independent_public_preprocessing=True, cells=35384, original_gene_features=23418, PC_dimensions=5,
    full_days=config['day_counts'], sparse_expression_nonzeros=report['artifacts'][0]['bytes'] // 8,
    original_preprocessing_configuration=config['original_preprocessing'], packaged_loader_configuration=package.CONFIG,
    input_npz=config['input_npz'], input_sha256=safe['input_sha256'], arrays=arrays,
    exact_original_author_loader_verified=True, numerical_PC_labels_and_source_rows_unchanged=True,
    complete_diagnostic_whitening_independently_checked=True, author_whiten_must_remain_true=True,
    pca_projection_relative_residual=report['pca_projection_relative_residual'], pca_orthogonality_error=report['pca_orthogonality_error'],
    original_inputs_overwritten=False, historical_paper_artifact_identity_certified=False,
    strict_whole_reference_gate_still_failed=True, original_paper_target_cell_roster_known=False,
    new_training_or_model_inference_executed=False, original_scope=config['original_scope'], all_original_experiments_complete=False)
with (folder / 'completed_input_validation.json').open('x', encoding='utf-8', newline='\n') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False); f.write('\n')
shutil.copyfile(__file__, folder / 'publication_source.py')
(folder / '.gitattributes').write_text('*.py -text\n*.json -text\n', encoding='utf-8', newline='\n')
(folder / 'README.md').write_text('''Complete independent public paper-described Multiome donor13176 input is ready locally. All35384 cells,23418 original genes and days2/3/4/7 (9546/8965/9407/7466 cells) were retained;137452310 nonzero expression coordinates were processed. Full five-dimensional centered PCA, original-author whitening block and original-author NPZ loader checks passed under the actual successful resource guard. PCA relative projection residual4.1499071472010065e-15; component orthogonality error1.9984014443252818e-15.

The packaged NPZ loads every array with allow_pickle=False. Original PC/label/source-row numeric arrays are unchanged; cell/gene identifiers are Unicode. All original preprocessing artifacts remain intact. The original author data module must keep whiten=True: this input contains unwhitened PCs; the whitened file is diagnostic only.

This is independent reconstruction under the paper-described cohort/preprocessing conditions. The strict historical whole-reference identity gate remains failed:4 author reference cells remain unresolved. Historical target cell roster and original PCA artifact identity are unknown. No training, new model inference, performance score, author-equivalence or strict-inductive TSAD result is claimed. Raw data and artifacts stay on the configured local/F: data storage; this packet stores evidence and checksums. Full scope remains45 papers/681 method-baseline-ablation records/273 data tasks/702 paper scalar targets. Original single-cell method/baseline/ablation experiments remain required.
''', encoding='utf-8', newline='\n')
targets = [BASE, package.MODULE.replace('.', '/') + '.py', package.TEST, package.CONFIG]
paths = set()
for args in [('diff', '--name-only'), ('ls-files', '--others', '--exclude-standard')]:
    paths.update(p for p in subprocess.check_output(['git', *args, '--', *targets], cwd=ROOT, text=True).splitlines() if not p.endswith('.log'))
assert paths and not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT).strip()
manifest = dict(previous_root=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    previous_parent=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT.parent / 'flow-matching-research', text=True).strip(), paths=sorted(paths))
(ROOT / 'tmp/OTCFM13176_completed_independent_input_exact_publication_20261016_v1.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(exact_files=len(paths), all_cells=35384, original_genes=23418, input_dimensions=5,
    independent_input_ready=True, strict_historical_identity_certified=False, no_training_performed=True)))
