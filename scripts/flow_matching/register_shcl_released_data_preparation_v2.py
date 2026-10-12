"""Freeze a new independent-reference audit version after verified v1 failure."""
import json

from scripts.flow_matching.prepare_shcl_released_data_v1 import native
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    predecessor='configs/datasets/fm_shcl_released_data_views.v1.json'
    config=native.read(native.ROOT/predecessor)
    base=native.ROOT/config['state_root']; receipt=native.read(base/'receipt.json')
    if receipt['exit_code']!=1 or receipt['resource_guard_aborted'] or receipt['passed']:
        raise ValueError('Exact naturally failed independent audit, not a capacity failure, required')
    error=(base/'worker.stderr.log').read_text(encoding='utf-8')
    if 'Original boundary window or source label slicing differs' not in error:
        raise ValueError('Exact identified v1 independent THRE-label reference failure required')
    source=native.ROOT/native.read(native.ROOT/config['model_config'])['source_root']/'data_factory/data_loader.py'
    if 'self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]' not in source.read_text(encoding='utf-8'):
        raise ValueError('Actual unchanged native nonoverlap label slicing required')
    extra=[predecessor,'scripts/flow_matching/prepare_shcl_released_data_v2.py',
        'scripts/flow_matching/run_shcl_released_data_preparation_v2.py',
        'scripts/flow_matching/register_shcl_released_data_preparation_v2.py',
        (base/'receipt.json').relative_to(native.ROOT).as_posix(),
        (base/'worker.stderr.log').relative_to(native.ROOT).as_posix()]
    config['source_receipts'].extend(dict(path=p,sha256=native.sha(native.ROOT/p)) for p in extra)
    config.update(state_root='experiments/runs/fm_shcl_released_data_preparation_v2',
        view_root='F:/aicoding/IIA_Data/experiments/shcl_released_native_v1/data_views_v2',
        independent_reference_correction='Only expected_label slicing in the independent loader audit is corrected. '
            'Native THRE returns validation features and test-prefix labels indexed by nonoverlap windows; '
            'these labels are not validation truth and native Solver ignores them when computing its threshold. '
            'All native source operations, complete inputs, output transforms,52 modes and guards are unchanged.',
        failed_v1_inputs_views_and_logs_preserved=True)
    exclusive(native.ROOT/'configs/datasets/fm_shcl_released_data_views.v2.json',config)
    print(json.dumps(dict(all_native_modes=52,correction='independent_THRE_label_reference_only',native_source_unchanged=True)))


if __name__=='__main__':main()
