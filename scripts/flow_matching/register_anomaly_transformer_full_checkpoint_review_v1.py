"""Freeze all500 reconstruction checkpoints for automatic full independent review."""
import json

from scripts.flow_matching import run_anomaly_transformer_full_checkpoint_review_v1 as runner

ROOT, read, sha = runner.ROOT, runner.read, runner.sha
CONFIG = 'configs/runtime/fm_anomaly_transformer_full_checkpoint_review.v1.json'


def registration():
    base = read(ROOT/runner.FAMILIES['main'][0])
    files = [v[0] for v in runner.FAMILIES.values()]
    for family_path, _, _ in runner.FAMILIES.values():
        files += [p['path'] for p in read(ROOT/family_path)['source_receipts']]
    files += [
        'scripts/flow_matching/anomaly_transformer_independent_math_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_full_checkpoint_review_v1.py',
        'scripts/flow_matching/register_anomaly_transformer_full_checkpoint_review_v1.py',
        'tests/test_fm_anomaly_transformer_full_checkpoint_review_v1.py',
        'scripts/flow_matching/heavy_scheduler_v2.py',
        'scripts/flow_matching/heavy_resources.py',
        'scripts/flow_matching/run_tsad_queue.py']
    return dict(schema_version=1, families={k: dict(configuration=v[0], required_models=v[1], module=v[2])
                for k, v in runner.FAMILIES.items()}, required_models=500,
        required_ids=[j['id'] for v in runner.FAMILIES.values() for j in read(ROOT/v[0])['jobs']],
        settings=base['settings'], python=base['python'], cpu_threads=2,
        resource_lock=base['resource_lock'], original_scope=base['original_scope'],
        formula_tolerance=dict(rtol=2e-6, atol=2e-6), metric_absolute_tolerance=2e-12,
        native_array_comparison='exact_dtype_shape_and_content_bytes',
        artifact_root='F:/aicoding/IIA_Data/experiments/anomaly_transformer_full_checkpoint_review_v1',
        state_root='experiments/runs/fm_anomaly_transformer_full_checkpoint_review_v1',
        source_receipts=[dict(path=p, sha256=sha(ROOT/p)) for p in sorted(set(files))],
        boundary='All25main+400single-factor+50direct-Max+25plain full paper reconstructions. '
            'Wait for actual complete training and passed original guards; restore the exact full checkpoint '
            'and independently re-infer every covered train/validation/test point. Compare all saved score '
            'and component arrays by dtype,shape,content bytes; independently check full-batch head/layer '
            'formulas in float64 at the frozen2e-6tolerance. Separate fresh child independently recalculates '
            'all validation-only thresholds,strict comparison,PA/noPA confusion counts,tied-rankAUROC/AP '
            'and all seven ROC operating points. Neither stage retrains or retries. Raw data and frozen '
            'sources unchanged; raw preprocessing values are not independently reconstructed. '
            'Original20runnable author-source+5restoredSWaT models remain separate and outside this500. '
            'No historical paper equivalence,strictTAB certification,uncertainty or full45/681/273completion claimed.')


def main():
    target = ROOT/CONFIG
    if target.exists():
        raise FileExistsError('Preserve the frozen full checkpoint review registration')
    config = registration(); runner.verify(config)
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(config, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(configuration=CONFIG, unique_models=500, numerical_workers_started=0)))


if __name__ == '__main__':
    main()
