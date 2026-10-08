"""Use the full author training/evaluation harness with explicit reconstructed rows."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'src'))
from iia_benchmark.models.fm_crossad_author_runtime import import_author
from iia_benchmark.models.fm_crossad_ablation import build_ablation
from iia_benchmark.models.fm_crossad_memory import install_activation_checkpointing
from scripts.flow_matching.run_crossad_author_job import run


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--queue', type=Path, required=True); parser.add_argument('--job-id', required=True)
    args = parser.parse_args(); queue = json.loads(args.queue.read_text()); settings = queue['settings']
    job = next(j for j in queue['jobs'] if j['id'] == args.job_id)
    module = import_author(ROOT/settings['runtime_root'])
    module.Exp_Anomaly_Detection._build_model = lambda self: install_activation_checkpointing(build_ablation(ROOT/settings['runtime_root'], job['model_parameters'], job['ablation_row']))
    run(job, settings)


if __name__ == '__main__':
    main()
