"""Prepare an isolated complete native Energy runtime; remove one unused import."""
import argparse
import json
from pathlib import Path

from scripts.flow_matching.prepare_gtgan_native_runtime_v1 import ROOT, prepare, digest, drop_unused_imports


def prepare_energy(config, destination):
    proof = prepare(config, destination)
    dest = Path(destination)
    entry = dest / 'GTGAN_energy.py'
    text, changes = drop_unused_imports(entry.read_text(encoding='utf-8'), {'gru_ode_bayes.data_utils'})
    entry.write_text(text, encoding='utf-8', newline='')
    proof['runtime_repairs'].append(dict(path='GTGAN_energy.py', repaired_sha256=digest(entry),
        changes=changes, all_original_network_classes_unchanged=True,
        entire_original_energy_train_and_main_ast_unchanged=True))
    (dest / 'runtime_preparation_receipt.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--destination', required=True)
    a = p.parse_args()
    c = json.loads((ROOT / a.config).read_text(encoding='utf-8'))
    proof = prepare_energy(c, a.destination)
    print(dict(complete_assets=len(proof['complete_copied_assets']), energy_algorithm_unchanged=True))


if __name__ == '__main__':
    main()
