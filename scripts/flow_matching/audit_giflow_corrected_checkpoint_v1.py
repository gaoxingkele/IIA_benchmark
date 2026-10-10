"""Replay the full reviewed GiFlow branch from its actual selected checkpoint.

The released replay parser expects a final EMA application. The reviewed branch
already loads the validation checkpoint and calls eval directly. Extract its
entire original test block, retaining every native batch and Euler update.
"""
import ast
import copy

from scripts.flow_matching import audit_giflow_checkpoint_replay_v1 as base


def native_test_function_corrected(source):
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_experiment')
    assignments = {n.targets[0].id: n for n in function.body if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    start = function.body.index(assignments['losses_mae'])
    block = function.body[start:start+4]
    if ([n.targets[0].id for n in block[:3]] != ['losses_mae', 'losses_mse', 'losses_mape']
            or not isinstance(block[3], ast.With)):
        raise ValueError('Full corrected native test-loop boundary changed')
    mode, loaded = function.body[start-1], function.body[start-2]
    if ast.unparse(mode) != 'model.eval()' or not ast.unparse(loaded).startswith('model.load_state_dict(torch.load(early_stopping.path,'):
        raise ValueError('Actual validation-selected tested weights required')
    steps = [n.value.value for n in ast.walk(block[3]) if isinstance(n, ast.Assign)
             and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
             and n.targets[0].id == 'num_steps' and isinstance(n.value, ast.Constant)]
    if steps != [20]:
        raise ValueError('All twenty original corrected Euler steps required')
    arguments = ['model', 'test_loader', 'node_embed', 'edge_index', 'H_spatial', 'H_temporal', 'args', 'device', 'ema']
    generated = ast.FunctionDef(name='replay_native_test', args=ast.arguments(posonlyargs=[],
        args=[ast.arg(arg=a) for a in arguments], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=copy.deepcopy([mode]+block), decorator_list=[])
    return ast.fix_missing_locations(ast.Module(body=[generated], type_ignores=[]))


def corrected_boundary(proof, result):
    if (result['job']['track'] != 'reviewed_corrected'
            or result['selection']['test_used_in_validation']
            or result['selection']['tested_weights_match_validation_checkpoint'] is not True
            or result['data_binding']['evaluation_targets_in_input'] != 0
            or result['data_binding']['timestamp_intersections'] != 0
            or not proof['full_checkpoint_contents_and_reinference_passed']):
        raise ValueError('Full reviewed checkpoint and clean validation/data bindings required')
    reviewed = copy.deepcopy(proof)
    reviewed.update(track='reviewed_corrected', selection=result['selection'],
        actual_full_training_epochs=result['epochs_completed'], actual_optimizer_updates=result['optimizer_updates'],
        tested_checkpoint_selected_on_validation=True, test_used_in_validation=False,
        native_replay_parser='scripts/flow_matching/audit_giflow_corrected_checkpoint_v1.py',
        source_test_block_unchanged=True, additional_training_seed_slots=0,
        boundary='Actual reviewed-source full validation-selected checkpoint with persistent EMA training, direct checkpoint load before eval. Full original20step Euler test block unchanged; batch128/last83, all1747 windows/268613 masked targets/280 forwards. CPU/GPU predictions and metrics checked under frozen existing tolerances; no bitwise prediction claim. No test-in-validation or per-batch EMA reset in this reviewed branch. Native overlapping-window batch-mean metrics retained; one of five original seeds, no original-paper/TAB equivalence certification.')
    return reviewed


def worker(config, config_path):
    base.worker(config, config_path)
    queue = base.read(base.ROOT / config['queue'])
    job = next(j for j in queue['jobs'] if j['id'] == config['job_id'])
    result = base.read(base.ROOT / job['output_directory'] / 'result.json')
    path = base.ROOT / config['output_directory'] / 'checkpoint_replay_audit.json'
    proof = corrected_boundary(base.read(path), result)
    proof.update(original_raw_replay_proof=str(path), original_raw_replay_proof_sha256=base.sha(path),
                 metadata_boundary_clarification_only=True)
    base.write(path.parent / 'reviewed_corrected_checkpoint_replay_audit.json', proof)


def main():
    # Parent/child entry binding only. Model, data and sampler execute unchanged.
    base.native_test_function = native_test_function_corrected
    tree = ast.parse(__import__('pathlib').Path(base.__file__).read_text(encoding='utf-8'))
    definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    replacement = 'scripts.flow_matching.audit_giflow_corrected_checkpoint_v1'
    count = 0
    class ChildEntry(ast.NodeTransformer):
        def visit_Constant(self, node):
            nonlocal count
            if node.value == 'scripts.flow_matching.audit_giflow_checkpoint_replay_v1':
                count += 1
                return ast.copy_location(ast.Constant(replacement), node)
            return node
    definition = ChildEntry().visit(definition)
    if count != 1:
        raise ValueError('Frozen full replay child-entry boundary changed')
    namespace = dict(vars(base), worker=worker)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])), base.__file__, 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
