"""Locate the suboperation inside the observed native RNG restoration failure."""
import ast
import inspect
from types import SimpleNamespace

from scripts.flow_matching import audit_shcl_cuda_rng_api_v2 as stages


def helper_ast():
    source = stages.original.job_api.replay.restore_and_verify_draws
    tree = ast.parse(inspect.getsource(source))
    snippets = {
        "original.set_backend_flags(checkpoint['backend_flags'])": 'native_backend_flag_restore',
        'original.restore_rng(state)': 'native_saved_RNG_restore',
        'original.verify_rng(state)': 'native_saved_RNG_readback',
        'expected_cpu = torch.rand(16, generator=cpu_generator)': 'independent_native_CPU_draw',
        'actual_cpu = torch.rand(16)': 'global_native_CPU_draw',
        'generator = torch.Generator(device=device)': 'native_CUDA_generator_construction',
        'generator.set_state(saved)': 'native_CUDA_generator_state_restore',
        'expected_cuda = torch.rand(16, generator=generator, device=device)': 'independent_native_CUDA_draw',
        'actual_cuda = torch.rand(16, device=device)': 'global_native_CUDA_draw',
    }
    targets = {ast.dump(ast.parse(code).body[0]): phase for code, phase in snippets.items()}
    class Observations(ast.NodeTransformer):
        count = 0
        def generic_visit(self, node):
            super().generic_visit(node)
            for key in ('body', 'orelse', 'finalbody'):
                value = getattr(node, key, None)
                if not isinstance(value, list):
                    continue
                result = []
                for statement in value:
                    phase = targets.get(ast.dump(statement))
                    if phase:
                        result += ast.parse('observe(' + repr('before_' + phase) + ')').body
                        self.count += 1
                    result.append(statement)
                    if phase:
                        result += ast.parse('observe(' + repr('after_' + phase) + ')').body
                        self.count += 1
                setattr(node, key, result)
            return node
    change = Observations()
    tree = change.visit(tree)
    if change.count != 22:
        raise ValueError('Exactly22 observations surround the11 exact original native operations')
    return ast.fix_missing_locations(tree)


def worker(config, path):
    observe = stages.observer(config)
    replay = stages.original.job_api.replay
    bindings = dict(replay.__dict__, observe=observe)
    exec(compile(helper_ast(), __file__, 'exec'), bindings)
    isolated_replay = SimpleNamespace(**dict(replay.__dict__, restore_and_verify_draws=bindings['restore_and_verify_draws']))
    isolated_job = SimpleNamespace(**dict(stages.original.job_api.__dict__, replay=isolated_replay))
    namespace = dict(stages.original.__dict__, observe=observe, job_api=isolated_job)
    exec(compile(stages.worker_ast(), __file__, 'exec'), namespace)
    namespace['worker'](config, path)


def main():
    tree = ast.parse(inspect.getsource(stages.original.main))
    changes = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == 'scripts.flow_matching.audit_shcl_cuda_rng_api_v1':
            node.value = 'scripts.flow_matching.audit_shcl_cuda_rng_api_v3'
            changes += 1
    if changes != 1:
        raise ValueError('Only the isolated observed worker entry changes')
    namespace = dict(stages.original.__dict__, worker=worker)
    exec(compile(ast.fix_missing_locations(tree), __file__, 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
