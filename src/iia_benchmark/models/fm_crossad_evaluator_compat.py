"""Defer only unused PATE imports in a new immutable CrossAD runtime copy."""
import json
from pathlib import Path

from iia_benchmark.models.fm_crossad_author_runtime import digest


def patch_evaluator(text):
    import_line = 'from . import pate\n'
    call = '        pate.evaluate(results_storage, metrics, labels=self.gt, score=self.anomaly_score, **metrics_args)\n'
    if text.count(import_line) != 1 or text.count(call) != 1:
        raise ValueError('Pinned PATE optional-evaluation anchors changed')
    return text.replace(import_line, '').replace(call,
        "        if 'pate' in metrics:\n            from . import pate\n" + '    ' + call)


def prepare_runtime(parent, target):
    parent, target = Path(parent), Path(target)
    manifest = json.loads((parent / 'runtime_manifest.json').read_text(encoding='utf-8'))
    records = []
    for entry in manifest['files']:
        source = parent / entry['path']
        assert digest(source) == entry['runtime_sha256'], entry['path']
        text = source.read_text(encoding='utf-8')
        patches = []
        if entry['path'] == 'ts_ad_evaluation/evaluator.py':
            text = patch_evaluator(text)
            patches = ['Import and invoke PATE only when the original metrics list requests pate']
        destination = target / entry['path']
        payload = text.encode('utf-8')
        if destination.exists() and destination.read_bytes() != payload:
            raise ValueError('Compatibility runtime already frozen differently')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(payload)
        records.append({'path': entry['path'], 'parent_sha256': entry['runtime_sha256'],
                        'runtime_sha256': digest(destination), 'patches': patches})
    receipt = {'parent_manifest_sha256': digest(parent / 'runtime_manifest.json'), 'files': records,
               'algorithm_changed': False, 'requested_metrics_changed': False,
               'pate_still_supported_when_requested': True,
               'boundary': 'Existing CPU mask placement patches retained. Only eager unused PATE import/call becomes conditional. '
                           'No error is swallowed if PATE itself is requested; its dependency compatibility remains separate work.'}
    payload = json.dumps(receipt, indent=2) + '\n'
    path = target / 'runtime_manifest.json'
    if path.exists() and path.read_text(encoding='utf-8') != payload:
        raise ValueError('Compatibility runtime manifest changed')
    path.write_text(payload, encoding='utf-8', newline='\n')
    return receipt
