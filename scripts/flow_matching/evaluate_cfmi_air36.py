"""Retain released Air-36 scores and separately correct its time cutoff axis."""
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.audit_author_code import patch_text
from scripts.flow_matching.metrics import apply_time_cutoffs


def adapt_source(original):
    adapted = patch_text(original, '    imputable_points = (~M_test) ^ (~Mtrue_test)\n',
                         '    imputable_points = (~M_test) ^ (~Mtrue_test)\n    corrected_imputable_points = imputable_points.copy()\n')
    adapted = patch_text(adapted, '    mean_scaler = None\n',
        '    if hparams.use_eval_cutoff:\n        corrected_imputable_points = apply_time_cutoffs(corrected_imputable_points, test_data[hparams.eval_cutoff_data_tensor_idx])\n\n    mean_scaler = None\n')
    adapted = patch_text(adapted, '                                         scaler=scaler, mean_scaler=mean_scaler)\n',
        '                                         scaler=scaler, mean_scaler=mean_scaler)\n        corrected_loss = compute_imputation_metric(X_test_copy, ~corrected_imputable_points, X_test_imp_copy, metric_name=metric, scaler=scaler, mean_scaler=mean_scaler)\n')
    adapted = patch_text(adapted, '                 loss=loss)\n',
        '                 loss=loss)\n        np.savez(os.path.join(experiment_dir, f"corrected_{metric}.npz"), loss=corrected_loss)\n')
    return adapted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.campaign_config.read_text(encoding='utf-8'))
    source = ROOT / cfg['author_source']
    sys.path.insert(0, str(source))
    original = (source / 'eval_imputation_timeseries.py').read_text(encoding='utf-8')
    adapted = adapt_source(original)
    output = ROOT / cfg['output_root']
    output.mkdir(parents=True, exist_ok=True)
    (output / 'evaluation_time_cutoff.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True), adapted.splitlines(True), fromfile='author/evaluation.py', tofile='dual-cutoff/evaluation.py')), encoding='utf-8')
    module = types.ModuleType('cfmi_air36_dual_evaluation')
    exec(compile(adapted, str(source / 'eval_imputation_timeseries.py'), 'exec'), module.__dict__)
    module.apply_time_cutoffs = apply_time_cutoffs
    module.parser = module.build_argparser()
    hparams = module.parser.parse_args([
        '--config', str(source / cfg['author_eval_config']),
        '--data.dataset.init_args.root', str(ROOT / cfg['data_root']),
        '--cfm_model_path', str(args.checkpoint.resolve()),
        '--default_root_dir', str(output), '--experiment_subdir_base', 'evaluation',
        '--num_imputations', str(cfg['num_samples']),
    ])
    module.run(hparams)


if __name__ == '__main__':
    main()
