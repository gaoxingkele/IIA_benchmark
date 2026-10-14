"""Dataset-labelled full native artifact review, using the frozen v1 checks."""
import argparse

from scripts.flow_matching.audit_spectral_completed_generation_v1 import audit
from scripts.flow_matching.prepare_spectral_original_data import ROOT, read, sha, write


def review(config):
    proof = audit(config)
    queue = read(ROOT / config['queue'])
    job = next(j for j in queue['jobs'] if j['id'] == config['job_id'])
    proof.update(model=job['model'], dataset=job['dataset'], data_track=job['data_track'],
        full_native_check_implementation='scripts/flow_matching/audit_spectral_completed_generation_v1.py',
        boundary=(f"Full original {job['model']} {job['dataset']} seed{job['seed']} "
            "native training, sampling and all four five-repeat evaluators completed. "
            "The unchanged v1 CPU audit verifies every model/EMA/optimizer tensor, "
            "complete generated/reference array, source hash and evaluator statistic. "
            "Five evaluator repetitions describe one trained model. No learned evaluator "
            "is retrained and no generative output is reinferred. Original reconstruction, "
            "all-source scaling and evaluator runtime boundaries remain; no future-held-out, "
            "author-equivalence or TAB claim. Full registered scope remains unfinished."))
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(ROOT / args.config)
    target = ROOT / config['output_directory']
    if target.exists():
        raise FileExistsError('Preserve immutable full generation review')
    proof = review(config)
    target.mkdir(parents=True)
    write(target / 'independent_full_native_artifact_audit.json', proof)
    write(target / 'validation.json', dict(passed=True,
        source_receipts=config['source_receipts'] + [dict(path=args.config, sha256=sha(ROOT / args.config))],
        outputs={'independent_full_native_artifact_audit.json': sha(target / 'independent_full_native_artifact_audit.json')}))
    print(dict(passed=True, id=proof['canonical_id'], training_models=1,
        metrics={key: value['mean'] for key, value in proof['metrics'].items()}))


if __name__ == '__main__':
    main()
