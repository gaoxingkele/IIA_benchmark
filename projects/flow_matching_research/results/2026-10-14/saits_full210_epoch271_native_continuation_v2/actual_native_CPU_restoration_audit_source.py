import hashlib
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
from scripts.flow_matching import run_saits as runner

root = runner.ROOT
configpath = root / 'configs/experiments/fm_saits_timeseries_jobs/saits_paper_brits_electricity_author_missing0.2_seed27.json'
config = json.loads(configpath.read_text())
output = root / config['output_root']
statepath, bestpath = output / 'resume.pt', output / 'best.pt'
state = torch.load(statepath, map_location='cpu', weights_only=False)
best = torch.load(bestpath, map_location='cpu', weights_only=False)
assert state['config_sha256'] == runner.sha(configpath) and state['dataset_sha256'] == config['dataset_sha256']
assert state['next_epoch'] == 271 and state['controller']['epoch'] == 271 and not state['finished']
assert state['controller']['train_step'] == 1897 and state['controller']['val_step'] == 271
source = root / 'experiments/runs/flow_matching_campaign/sources/saits/corrected'
sys.path.insert(0, str(source))
import run_models as author
args, parameters = runner.configure(author, config, output)
assert args.epochs == 10000 and args.early_stop_patience == 30
parameters['device'] = 'cpu'
model = author.MODEL_DICT[args.model_type](**parameters).to('cpu')
model.load_state_dict(state['model'], strict=True)
for name, actual in model.state_dict().items():
    assert torch.equal(actual, state['model'][name]) and torch.isfinite(actual).all()
optimizer = author.OPTIMIZER[args.optimizer_type](model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer.load_state_dict(state['optimizer'])
restored = optimizer.state_dict()
assert restored['param_groups'] == state['optimizer']['param_groups']
assert len(restored['state']) == len(state['optimizer']['state']) == 28
for index, fields in restored['state'].items():
    for name, actual in fields.items():
        expected = state['optimizer']['state'][index][name]
        if isinstance(actual, torch.Tensor):
            assert torch.equal(actual, expected) and torch.isfinite(actual).all()
        else: assert actual == expected
    assert float(fields['step']) == 1897
random.setstate(state['python_rng']); assert random.getstate() == state['python_rng']
np.random.set_state(state['numpy_rng'])
actual = np.random.get_state(); expected = state['numpy_rng']
assert actual[0] == expected[0] and np.array_equal(actual[1], expected[1]) and actual[2:] == expected[2:]
torch.set_rng_state(state['torch_rng'].cpu()); assert torch.equal(torch.get_rng_state(), state['torch_rng'].cpu())
assert len(state['cuda_rng']) == 1 and all(x.dtype == torch.uint8 and x.numel() > 0 for x in state['cuda_rng'])
assert float(best['validation_mae']) == float(state['controller']['best_imputation_MAE'])
proof = dict(canonical_id=config['id'], config_sha256=runner.sha(configpath), dataset_sha256=state['dataset_sha256'],
    full_saved_epoch_boundary=271, saved_optimizer_updates=1897, finished=False,
    full_native_model_tensors_restored_bitwise=len(model.state_dict()),
    full_native_optimizer_parameter_slots_restored_bitwise=len(restored['state']),
    every_optimizer_step_equals1897=True, Python_NumPy_Torch_CPU_RNG_restored_exactly=True,
    saved_CUDA_RNG_states=1, CUDA_RNG_runtime_restore_verified=False,
    checkpoint_path=str(statepath), checkpoint_sha256=runner.sha(statepath),
    best_checkpoint_path=str(bestpath), best_checkpoint_sha256=runner.sha(bestpath),
    best_validation_MAE=float(best['validation_mae']), original_max_epochs=10000, original_early_stop_patience=30,
    native_checkpoint_continuation_code='scripts/flow_matching/run_saits.py',
    native_checkpoint_continuation_code_sha256=runner.sha(root/'scripts/flow_matching/run_saits.py'),
    numeric_optimizer_updates_executed=0, full_training_or_test_complete=False, author_equivalence_certified=False)
folder = root / 'projects/flow_matching_research/results/2026-10-14/saits_full210_epoch271_native_continuation_v2'
folder.mkdir(parents=True, exist_ok=True)
(folder/'actual_native_CPU_epoch271_restoration_proof.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
print(json.dumps(proof))
