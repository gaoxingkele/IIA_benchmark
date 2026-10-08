"""Bind unchanged saved range metrics to a new, fully cited execution config."""
import copy
import hashlib
import json
from pathlib import Path

from scripts.flow_matching.prepare_tsad_execution import sha, write_json
from scripts.flow_matching.run_tsad_queue import complete_result


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def reuse_completed(root, result_path, config):
    result = json.loads(result_path.read_text(encoding='utf-8'))
    if not complete_result(root, result['job']):
        raise ValueError('Model result is incomplete')
    parent_path = root / config['parent_config']
    if sha(parent_path) != config['parent_config_sha256']:
        raise ValueError('Parent range config changed')
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    old_path = root / parent['output_root'] / 'jobs' / result['id'] / 'evaluation.json'
    if not old_path.exists():
        return None
    old_receipt = old_path.with_name('evaluation.sha256')
    if not old_receipt.exists():
        return None  # A valid old producer may still be finishing its atomic writes.
    if old_receipt.read_text(encoding='ascii').strip() != sha(old_path):
        raise ValueError('Original completed range evaluation changed')
    old = json.loads(old_path.read_text(encoding='utf-8'))
    result_sha = sha(result_path)
    parent_identity = hashlib.sha256((result_sha + config_hash(parent)).encode()).hexdigest()
    if old['evaluation_identity'] != parent_identity or old['model_result_sha256'] != result_sha:
        raise ValueError('Original range evaluation belongs to different rules/data')
    if old['scores_sha256'] != result['scores_sha256']:
        raise ValueError('Original score identity changed')
    final_path = root / config['output_root'] / 'jobs' / result['id'] / 'evaluation.json'
    identity = hashlib.sha256((result_sha + config_hash(config)).encode()).hexdigest()
    if final_path.exists():
        record = json.loads(final_path.read_text(encoding='utf-8'))
        if record['evaluation_identity'] != identity or final_path.with_name('evaluation.sha256').read_text(encoding='ascii').strip() != sha(final_path):
            raise ValueError('New range binding changed')
        return record
    record = copy.deepcopy(old)
    record.update(evaluation_identity=identity, evaluation_config_sha256=config_hash(config), source_receipts=config['source_receipts'],
                  reused_original_evaluation={'path': old_path.relative_to(root).as_posix(), 'sha256': sha(old_path),
                                              'all_metric_values_and_original_elapsed_time_preserved': True},
                  execution_engine='verified_original_result_rebound_without_recomputation')
    write_json(final_path, record)
    final_path.with_name('evaluation.sha256').write_text(sha(final_path) + '\n', encoding='ascii')
    return record
