"""Prepare exact publication paths for two completed original model reviews."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from scripts.flow_matching import run_pi_causal1105_completed_full_replay_v1 as replay

root = replay.ROOT
config = replay.read('configs/runtime/fm_pi_causal1105_completed_full_replay.v1.json')
replay.verify(config)
pi_base = 'projects/flow_matching_research/results/2026-10-14/pi_causal1105_actual_complete_model_replay_closure_v1'
br_base = 'projects/flow_matching_research/results/2026-10-14/brits_electricity30_seed27_closed_full_replay_v1'
pi = replay.read(pi_base + '/one_complete_original_model_and_all_fifteen_streams_closure.json')
br = replay.read(br_base + '/actual_original_brits_electricity30_seed27_complete_closure.json')
assert pi['selected_complete_models'] == 1 and pi['all_streams_bitwise_identical_to_original_saved_arrays']
assert pi['complete_original_score_streams'] == 15
assert pi['inherited_proof_prose_mismatch_disclosure']
assert br['full_original_training_and_model_reinference_and_independent_array_review_passed']
assert br['verified_SAITS_BRITS_models'] == 36 and br['completed_verified_seeds_of_this_rate'] == [26,27]
assert not br['five_seed_aggregate_complete'] and not br['all_original_experiments_complete']
xml_path = root / 'tmp/Pi_causal1105_BRITS27_full36_closure_validation_20261014_v1.xml'
suites = list(ET.parse(xml_path).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites) == 97
assert all(int(s.get('failures',0)) == int(s.get('errors',0)) == 0 for s in suites)
folder = root / 'projects/flow_matching_research/results/2026-10-14/pi_causal1105_and_brits27_full36_increment_v1'
folder.mkdir(exist_ok=False)
source_paths = [pi_base + '/one_complete_original_model_and_all_fifteen_streams_closure.json',
                br_base + '/actual_original_brits_electricity30_seed27_complete_closure.json']
summary = dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    newly_closed_original_models=2, original_scope=br['original_scope'],
    Pi=dict(canonical_id=replay.CANONICAL, task='time_series_anomaly_detection',
        original_full_epochs=3, optimizer_minibatches=1554, parameters=1930289,
        complete_bitwise_replayed_streams=15, original_train_score_points=13238200,
        original_covered_test_points=87800, original_dropped_test_tail_points=41,
        original_required_jobs=1140, original_required_training_seeds=[42,1103,1104,1105,1106,1107],
        metrics=pi['independently_recomputed_metrics'],
        source_test_selection_and_pooled_test_calibration_retained=True,
        inherited_template_prose_mismatch=pi['inherited_proof_prose_mismatch_disclosure']),
    BRITS=dict(canonical_id=br['id'], task='time_series_imputation',
        original_maximum_epochs=10000, next_epoch=940, best_epoch_source_index=908,
        original_author_patience=30, parameters=6993328, held_out_coordinates=2781282,
        full_test_shape=[290,100,370], original_month_groups=10,
        independently_recomputed_metrics=br['independently_recomputed_metrics'],
        conditional_month_block_ci95=br['original_month_block_bootstrap_95_ci'],
        completed_seeds_at_this_rate=[26,27], required_seeds_at_this_rate=[26,27,28,29,30],
        full_SAITS_BRITS_models_verified=36, full_SAITS_BRITS_models_required=210,
        five_seed_aggregate_complete=False, bitwise_original_prediction_equivalence_claimed=False),
    validation_tests_passed=97,
    source_receipts=[dict(path=p,sha256=replay.digest(p)) for p in source_paths],
    historical_author_equivalence_certified=False,strict_TAB_result=False,
    all_original_experiments_complete=False)
(folder/'two_complete_original_model_reviews.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
(folder/'publication_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
(folder/'validation.xml').write_bytes(xml_path.read_bytes())
(folder/'README.md').write_text(
    'Two additional original model reviews are closed: Pi causal-axis PSM seed1105 '
    'and BRITS Electricity nominal30% seed27. All original budgets, guards and tolerances remain. '
    '97 relevant validation tests passed.\n\n'
    'Pi: all15 checkpoint-reinferred streams match the source arrays bitwise across '
    '13,238,200 training score points and87,800 originally covered test points. '
    'PA F1=0.9788686340 versus noPA F1=0.0191087853 at the same source threshold; '
    'test-based selection, pooled train/test calibration and41 dropped tail points remain. '
    'Frozen review template prose incorrectly names source_runtime seed42; actual structured identity '
    'and all checkpoint/config/result/array digests bind causal_axis_repaired seed1105. '
    'The original proof is preserved unchanged.\n\n'
    'BRITS: full10000-epoch maximum/patience30 protocol terminated at next_epoch940, '
    'best source index908. Full290x100x370 grid /2,781,282 held-out coordinates /10 months '
    'passed native checkpoint replay and independent metric review at unchanged2e-6 tolerance. '
    'MAE=0.9764629584;RMSE=1.4681269702;MRE=0.5226215631. '
    'Only seeds26,27 are closed at this rate; no five-seed aggregate is claimed. '
    'Overall SAITS/BRITS complete reviewed models:36/210.\n\n'
    'These cases address different tasks and do not form a shared algorithm ranking. '
    'Full1140 Pi jobs/six seeds,210 SAITS/BRITS jobs and45-paper/681-method/273-data scope remain. '
    'No historical-author or strictTAB certificate; all original experiments remain incomplete.\n',encoding='utf-8')
targets = [pi_base,br_base,folder.relative_to(root).as_posix(),
    'projects/flow_matching_research/results/2026-10-14/pi_causal1105_completed_full_replay_v1',
    'configs/reproducibility/fm_pi_causal1105_completed_full_replay_children_v1',
    'configs/reproducibility/fm_saits_completion_replay_queue_v1/saits_paper_brits_electricity_author_missing0.3_seed27.native.json',
    'configs/reproducibility/fm_saits_completion_replay_queue_v1/saits_paper_brits_electricity_author_missing0.3_seed27.review.json',
    'projects/flow_matching_research/results/2026-10-11/saits_automatic_complete_model_reviews_v1/saits_paper_brits_electricity_author_missing0.3_seed27']
def git_files(*args):
    raw = subprocess.check_output(['git','-c','core.longpaths=true',*args,'--',*targets],cwd=root)
    return set(raw.decode('utf-8').splitlines())
paths = {path for path in git_files('diff','--name-only') | git_files('ls-files','--others','--exclude-standard')
         if not path.endswith('.log')}
assert paths and all(not path.endswith('.log') for path in paths)
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root).strip()
manifest=dict(previous_root='2e0fff629be09a0d30faec61aa04b9ec480797ae',
    previous_parent='d727d450156e744d78090decf2728944f4bfd18e',paths=sorted(paths))
(root/'tmp/Pi_causal1105_BRITS27_full36_publication_manifest_20261014_v1.json').write_text(
    json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(exact_files=len(paths),original_models_closed=2,validation_tests_passed=97,
    full_SAITS_BRITS_models_verified=36,all_original_experiments_complete=False)))
