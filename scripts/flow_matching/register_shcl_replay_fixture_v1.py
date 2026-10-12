"""Pin every saved diagnostic state and full score frame before independent replay."""
import json
from pathlib import Path

from scripts.flow_matching import shcl_released_native_v1 as native
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    predecessor='configs/runtime/fm_shcl_native_observer_audit.v1.json'; config=native.read(native.ROOT/predecessor)
    fixture=Path('F:/aicoding/IIA_Data/experiments/shcl_released_native_v1/diagnostics/observer_fixture_v1')
    inputs=[fixture/'training_observer_fixture.json',fixture/'test_observer_fixture.json',fixture/'native_checkpoint/SMAP_checkpoint.pth']
    training=native.read(inputs[0]); test=native.read(inputs[1]); paths=[str(p) for p in inputs]
    paths.extend(row['full_state'] for row in training['epochs'])
    paths.extend(value for row in test['full_source_score_frames'] for value in (row['context'],row['arrays']))
    paths.extend([predecessor,'scripts/flow_matching/replay_shcl_native_frames_v1.py',
        'scripts/flow_matching/audit_shcl_replay_fixture_v1.py','scripts/flow_matching/run_shcl_replay_fixture_v1.py',
        'scripts/flow_matching/register_shcl_replay_fixture_v1.py'])
    config['source_receipts'].extend(dict(path=p,sha256=native.sha(native.ROOT/p)) for p in paths)
    config.update(fixture_training=str(inputs[0]),fixture_test=str(inputs[1]),fixture_checkpoint=str(inputs[2]),
        state_root='experiments/runs/fm_shcl_replay_fixture_v1',
        output_root='projects/flow_matching_research/results/2026-10-12/shcl_independent_replay_preparation_v1/fixture',
        replayed_array_root='F:/aicoding/IIA_Data/experiments/shcl_released_native_v1/diagnostics/independent_replay_fixture_v1',
        full_native_frames_required=len(test['full_source_score_frames']),
        full_native_windows_required=sum(row['shape'][0] for row in test['full_source_score_frames']),
        bitwise_array_and_post_RNG_gate=True,
        boundary='Independent all-state/all-frame diagnostic replay. Every original fixture model/Adam '
            'variable, full threshold/test loss/score/label array and post-RNG stream must agree exactly. '
            'No tolerance is loosened; no original source training or benchmark performance is credited. '
            'All49 original trainings and364 tests remain separate obligations.')
    exclusive(native.ROOT/'configs/runtime/fm_shcl_independent_replay_fixture.v1.json',config)
    print(json.dumps(dict(full_checkpoints=len(training['epochs']),complete_frames=len(test['full_source_score_frames']),
        complete_windows=config['full_native_windows_required'],new_benchmark_results=0)))


if __name__=='__main__':main()
