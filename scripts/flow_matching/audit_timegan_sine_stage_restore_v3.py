"""Full Sine checkpoint restoration and native transition audit, not training.

The v2 audit incorrectly routes sine to the real-data-only author loader.
Only that loader expression is replaced with the untouched author sine loader;
all original graph, variable/optimizer restoration and bitwise transitions stay.
"""
import ast
import inspect

from scripts.flow_matching import audit_timegan_stage_resume_v2 as original


def audit_ast():
    tree = ast.parse(inspect.getsource(original.audit))
    expected = ast.dump(ast.parse("real_data_loading(job['dataset'], model['parameters']['seq_len'])", mode='eval').body)
    replacement = ast.parse(
        "_native_sine_loader(job, model, np)", mode='eval').body
    matches = [0]

    class Loader(ast.NodeTransformer):
        def visit_Call(self, node):
            if ast.dump(node) == expected:
                matches[0] += 1
                return ast.copy_location(replacement, node)
            return self.generic_visit(node)

    tree = Loader().visit(tree)
    if matches != [1]:
        raise ValueError('Exactly one full native data loader expression may change')
    return ast.fix_missing_locations(tree)


def native_sine_loader(job, model, np):
    if job['dataset'] != 'sine':
        raise ValueError('This exact full Sine audit must not alter real dataset loading')
    item = original.read(original.ROOT / model['dataset_config'])['datasets']['sine']
    from data_loading import sine_data_generation
    result = sine_data_generation(item['samples'], model['parameters']['seq_len'], item['features'])
    # Independently check every coordinate against the documented original generator.
    rng = np.random.RandomState(job['seed'])
    independent = []
    for _ in range(item['samples']):
        columns = []
        for _ in range(item['features']):
            frequency, phase = rng.uniform(0, .1), rng.uniform(0, .1)
            columns.append([np.sin(frequency * j + phase) for j in range(24)])
        independent.append((np.asarray(columns).T + 1) * .5)
    from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import bitwise_array
    bitwise_array(result, independent, np, 'all original full Sine loader coordinates')
    return result


def corrected_audit():
    namespace = dict(original.__dict__, _native_sine_loader=native_sine_loader)
    exec(compile(audit_ast(), original.__file__, 'exec'), namespace)
    return namespace['audit']


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = original.read(original.ROOT / args.config)
    destination = original.ROOT / config['native_restore_audit']
    if destination.exists():
        raise FileExistsError('Preserve every existing native restore audit')
    original.write(destination, corrected_audit()(config))


if __name__ == '__main__':
    main()
