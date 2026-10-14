"""Full saved-score audit for separate industrial train/validation/test groups.

Preserve the completed MOMENT auditor's weight, environment, metric, bootstrap,
range and resource checks. Adapt only the registered industrial array layout.
"""
import ast
import inspect

from scripts.flow_matching import audit_moment_completed_saved_scores_v1 as original

METADATA_SUFFIXES = ('_labels', '_timestamps', '_source_rows', '_fault_labels')


def groups(dataset):
    rows = [(role, entry, entry['array_prefix'] + ('_test' if role == 'test' else '_values'))
            for role, records in [('train', dataset['training_groups']),
                ('validation', dataset['validation_groups']), ('test', dataset['entities'])]
            for entry in records]
    if any(not records for records in (dataset['training_groups'], dataset['validation_groups'], dataset['entities'])):
        raise ValueError('Every original role must be present')
    prefixes = [entry['array_prefix'] for _, entry, _ in rows]
    if len(set(prefixes)) != len(prefixes):
        raise ValueError('Distinct frozen industrial array prefixes required')
    roles = {}
    for role, entry, _ in rows:
        group = entry['group_id']
        if group in roles and roles[group] != role:
            raise ValueError('Whole original runs cannot cross train/validation/test roles')
        roles[group] = role
    return rows


def check_industrial_arrays(inputs, saved, dataset):
    import numpy as np
    expected, entities, lengths, validation = set(), {}, {}, []
    for role, entry, input_key in groups(dataset):
        prefix = entry['array_prefix']
        values = inputs[input_key]
        key = prefix + '_' + role
        expected.add(key)
        if (values.ndim != 2 or values.shape[1] != entry['features']
                or len(values) != entry['points'] or not np.isfinite(values).all()
                or saved[key].shape != (len(values),) or not np.isfinite(saved[key]).all()):
            raise ValueError('Complete finite original features and timestamp scores required: ' + key)
        for suffix in METADATA_SUFFIXES:
            metadata = prefix + suffix
            if metadata in inputs:
                expected.add(metadata)
                if metadata not in saved or not np.array_equal(inputs[metadata], saved[metadata]):
                    raise ValueError('Original label/timestamp/source-row/fault metadata changed')
        if role == 'validation':
            validation.append(saved[key])
        elif role == 'test':
            labels = prefix + '_labels'
            entity = entry['entity_id']
            if (labels not in inputs or saved[labels].shape != (entry['test_points'],)
                    or entry['test_points'] != len(values) or entity in entities):
                raise ValueError('Each full original test entity and labels required')
            entities[entity] = (saved[labels], saved[key])
            lengths[entity] = len(values)
    if set(saved.files) != expected:
        raise ValueError('Every original score and copied metadata array must be present exactly')
    return entities, lengths, np.concatenate(validation)


def check_industrial_coverage(inputs, result, dataset):
    expected = {(entry.get('entity_id', entry['group_id']), role): (entry, input_key)
                for role, entry, input_key in groups(dataset)}
    rows = result['native_coverage']
    actual = {(row['group'], row['role']): row for row in rows}
    if len(actual) != len(rows) or set(actual) != set(expected):
        raise ValueError('Every distinct original industrial segment coverage required')
    for identity, (entry, input_key) in expected.items():
        row = actual[identity]
        if (row['source_array'] != input_key or row['native_points'] != len(inputs[input_key])
                or row['native_points'] != entry['points'] or not row['all_context_tokens_observed']):
            raise ValueError('Full original coverage and observed context must agree')


def worker_ast():
    tree = ast.parse(inspect.getsource(original.worker))
    changes = dict(import_layout=0, training_groups=0, coverage=0)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'scripts.flow_matching.verify_tranad_precision_and_smd_completion_v1':
            assert [(alias.name, alias.asname) for alias in node.names] == [('check_arrays', None)]
            node.module = __name__ if __name__ != '__main__' else 'scripts.flow_matching.audit_moment_industrial_saved_scores_v1'
            node.names = [ast.alias(name='check_industrial_arrays', asname='check_arrays')]
            changes['import_layout'] += 1
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'training':
            assert ast.unparse(node.value) == "np.concatenate([scores[entry['array_prefix'] + '_train'] for entry in dataset['entities']])"
            node.value = ast.parse("np.concatenate([scores[entry['array_prefix'] + '_train'] for entry in dataset['training_groups']])", mode='eval').body
            changes['training_groups'] += 1
        if isinstance(node, ast.With):
            index = next((i for i, item in enumerate(node.body) if isinstance(item, ast.Assign)
                          and ast.unparse(item.targets[0]) == 'coverage'), None)
            if index is not None:
                block = node.body[index:index+3]
                assert len(block) == 3 and isinstance(block[1], ast.If) and isinstance(block[2], ast.For)
                assert ast.unparse(block[1].test) == "len(coverage) != len(dataset['entities']) * 3"
                assert ast.unparse(block[2].iter) == "dataset['entities']"
                node.body[index:index+3] = ast.parse('check_industrial_coverage(inputs, result, dataset)').body
                changes['coverage'] += 1
    if changes != dict(import_layout=1, training_groups=1, coverage=1):
        raise ValueError('Only three exact industrial input-layout boundaries may change')
    return ast.fix_missing_locations(tree)


def worker(config, config_path):
    values = dict(vars(original), check_industrial_coverage=check_industrial_coverage)
    exec(compile(worker_ast(), __file__, 'exec'), values)
    values['worker'](config, config_path)


def main():
    tree = ast.parse(inspect.getsource(original.main))
    changes = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == 'scripts.flow_matching.audit_moment_completed_saved_scores_v1':
            node.value = 'scripts.flow_matching.audit_moment_industrial_saved_scores_v1'
            changes += 1
    if changes != 1:
        raise ValueError('Only the industrial worker entry point may change')
    values = dict(vars(original), worker=worker)
    exec(compile(ast.fix_missing_locations(tree), __file__, 'exec'), values)
    values['main']()


if __name__ == '__main__':
    main()
