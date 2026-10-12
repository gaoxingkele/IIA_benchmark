"""Pin evidence of the RSS abort and eliminate only unused data-audit imports."""
import json

from scripts.flow_matching.prepare_shcl_released_data_v1 import native
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    predecessor='configs/datasets/fm_shcl_released_data_views.v2.json'; config=native.read(native.ROOT/predecessor)
    base=native.ROOT/config['state_root']; receipt=native.read(base/'receipt.json')
    if (receipt['passed'] or not receipt['resource_guard_aborted']
            or receipt['peak_tree_rss_bytes']<=config['settings']['maximum_worker_rss_bytes']):
        raise ValueError('Exact unchanged maximum-RSS guard abort required before import-only recovery')
    from scripts.flow_matching.prepare_shcl_released_data_v3 import data_module_ast,loader_audit_ast
    model=native.read(native.ROOT/config['model_config']); source=native.verify(model)/'data_factory/data_loader.py'
    data_module_ast(source.read_text(encoding='utf-8')); loader_audit_ast()
    extra=[predecessor,'scripts/flow_matching/prepare_shcl_released_data_v3.py',
        'scripts/flow_matching/run_shcl_released_data_preparation_v3.py',
        'scripts/flow_matching/register_shcl_released_data_preparation_v3.py',
        (base/'receipt.json').relative_to(native.ROOT).as_posix()]
    config['source_receipts'].extend(dict(path=p,sha256=native.sha(native.ROOT/p)) for p in extra)
    config.update(state_root='experiments/runs/fm_shcl_released_data_preparation_v3',
        view_root='F:/aicoding/IIA_Data/experiments/shcl_released_native_v1/data_views_v3',
        isolated_import_correction='Omit only unused import torch and from torch.utils.data import DataLoader '
            'while compiling the original native dataset module for the data-class audit. Every class body, '
            'full raw matrix, scaler operation,52 mode/profile obligations, boundary checks and guards unchanged. '
            'The native get_loader_segment factory is not called by this data audit. Full training still uses '
            'the unchanged complete module and its original8 DataLoader workers.',
        failed_v1_v2_views_logs_and_caps_preserved=True)
    exclusive(native.ROOT/'configs/datasets/fm_shcl_released_data_views.v3.json',config)
    print(json.dumps(dict(full_native_modes=52,unused_Torch_imports_removed=2,original_guards_unchanged=True)))


if __name__=='__main__':main()
