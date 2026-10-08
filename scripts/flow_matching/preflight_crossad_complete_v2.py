"""Unchanged 22 full-native-batch checks with resource-aware lock scheduling."""
from scripts.flow_matching import preflight_crossad_complete as controller
from scripts.flow_matching.heavy_scheduler_v2 import run_bound_controller

if __name__ == '__main__':
    run_bound_controller(controller)
