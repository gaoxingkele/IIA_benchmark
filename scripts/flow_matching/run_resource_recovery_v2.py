"""Unchanged incident retries with resource-aware shared-lock scheduling."""
from scripts.flow_matching import run_resource_recovery as controller
from scripts.flow_matching.heavy_scheduler_v2 import run_bound_controller

if __name__ == '__main__':
    run_bound_controller(controller)
