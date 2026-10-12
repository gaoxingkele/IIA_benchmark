"""Replay all actual diagnostic checkpoints/frames; no benchmark training credit."""
import argparse
from pathlib import Path

from scripts.flow_matching import replay_shcl_native_frames_v1 as replay
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def worker(config,config_path):
    for pin in config['source_receipts']:
        if replay.native.sha(replay.native.ROOT/pin['path'])!=pin['sha256']: raise ValueError('Frozen complete replay diagnostic changed')
    if config['environment']!=replay.environment(): raise ValueError('Exact acquired native environment required')
    import torch
    torch.set_num_threads(2)
    model=replay.native.read(replay.native.ROOT/config['model_config'])
    training=replay.native.read(config['fixture_training']); test=replay.native.read(config['fixture_test'])
    if not training['job'].get('synthetic_fixture') or test['job']!=training['job']:
        raise ValueError('Diagnostic inputs can never substitute for original-data source results')
    report=dict(passed=False,diagnostic_fixture_only=True,original_source_train49_and_test364_still_required=True)
    replay.replay_training(model,training,config['fixture_checkpoint'],report)
    replay.replay_test(model,test,config['fixture_checkpoint'],config['replayed_array_root'],report)
    report.update(passed=True,configuration=config_path,configuration_sha256=replay.native.sha(replay.native.ROOT/config_path),
        all_native_parameters_and_Adam_state_bitwise_unchanged_by_observation=True,
        independent_checkpoint_and_all_full_fixture_frame_replay_passed=True,original_benchmark_optimizer_updates=0)
    exclusive(replay.native.ROOT/config['output_root']/'full_observer_fidelity_fixture.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False); parser.add_argument('--config',required=True)
    args=parser.parse_args(); worker(replay.native.read(replay.native.ROOT/args.config),args.config)


if __name__=='__main__':main()
