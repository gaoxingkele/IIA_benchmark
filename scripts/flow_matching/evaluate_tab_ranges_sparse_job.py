"""One full saved-score evaluation; the child process releases all work arrays."""
import argparse
import json
from pathlib import Path

from scripts.flow_matching.prepare_tsad_execution import sha
from scripts.flow_matching.tab_affiliation_sparse import load_sparse_reference
from scripts.flow_matching.evaluate_tab_ranges import evaluate

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding='utf-8'))
    for source in config['source_receipts']:
        assert sha(ROOT / source['path']) == source['sha256'], source['path']
    reference = load_sparse_reference(ROOT / config['reference_metric_root'])
    evaluate(ROOT, ROOT / args.result, config, reference)


if __name__ == '__main__':
    main()
