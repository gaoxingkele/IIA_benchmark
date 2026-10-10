"""Keep all four full Table3 jobs; use registered full GPU memory admission."""
import argparse
import sys

from scripts.flow_matching import run_spectral_table3_queue_v2 as original
from scripts.flow_matching.full_gpu_memory_slot_v2 import full_gpu_memory_slot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    registration = original.read(original.ROOT / args.runtime_config)
    for source in registration['source_receipts']:
        if original.sha(original.ROOT / source['path']) != source['sha256']:
            raise ValueError('Frozen GPU memory dispatcher source differs')
    if original.sha(original.ROOT / registration['queue']) != registration['queue_sha256']:
        raise ValueError('Full four original seed slots changed')
    original.complete_native_slot = full_gpu_memory_slot
    sys.argv = ['full_table3', '--queue', registration['queue']] + (['--probe-only'] if args.probe_only else [])
    original.main()


if __name__ == '__main__':
    main()
