"""Source-preserving native recovery controller with the existing resource mutexes."""
import json
import sys
from scripts.flow_matching.native_exact_recovery import ROOT, controller_main


if __name__ == '__main__':
    path = sys.argv[sys.argv.index('--queue') + 1]
    queue = json.loads((ROOT / path).read_text(encoding='utf-8'))
    controller_main(queue['recovery_track'])()
