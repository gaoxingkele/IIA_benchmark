"""Full 139-case CrossAD controller with resource-aware shared-lock scheduling."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.flow_matching import run_crossad_case_gated_queue as controller
from scripts.flow_matching.heavy_scheduler_v2 import run_bound_controller

if __name__ == '__main__':
    run_bound_controller(controller)
