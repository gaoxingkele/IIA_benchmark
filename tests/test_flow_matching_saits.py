import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

from scripts.flow_matching.compare_saits import compare
from scripts.flow_matching.prepare_saits_physio import ROOT, extract_patients
from scripts.flow_matching.run_saits import patient_bootstrap


def test_bootstrap_weights_coordinates_and_keeps_patients_together():
    statistics = np.array([[1, 1, 1, 2], [90, 900, 9, 18]])
    result = patient_bootstrap(statistics, 26, repetitions=1000)
    assert result['mae'] == [1, 10]
    assert result['rmse'] == [1, 10]
    assert result['mre'] == [0.5, 5]


def test_extract_rejects_links_instead_of_resolving_arbitrary_paths(tmp_path):
    archive = tmp_path / 'bad.tar.gz'
    with tarfile.open(archive, 'w:gz') as handle:
        link = tarfile.TarInfo('../1.txt')
        link.type, link.linkname = tarfile.SYMTYPE, '../outside'
        handle.addfile(link)
    with pytest.raises(ValueError, match='Unexpected'):
        extract_patients([archive], tmp_path / 'patients', 1)


def test_registered_data_matches_author_paper_patient_protocol():
    directory = ROOT / 'data/public_datasets/flow_matching/processed_campaign_v1/author_saits/physio2012_37feats_01masked'
    if not directory.exists():
        pytest.skip('Optional downloaded real data')
    audit = json.loads((directory / 'data_audit.json').read_text(encoding='utf-8'))
    assert [audit['splits'][k]['patients'] for k in ('train', 'val', 'test')] == [7672, 1918, 2398]
    for split in ('val', 'test'):
        assert 0.094 < audit['splits'][split]['realized_missing_rate'] < 0.096
    with np.load(directory / 'audit_arrays.npz') as arrays:
        assert len(arrays['feature_names']) == 37
        assert 'Weight' in arrays['feature_names'] and 'MechVent' in arrays['feature_names']
        assert not set(arrays['train_ids']) & set(arrays['test_ids'])


def test_registered_queue_is_frozen_and_comparison_waits_for_all_seeds():
    path = ROOT / 'configs/experiments/fm_saits_physio_original_queue.v1.json'
    queue = json.loads(path.read_text(encoding='utf-8'))
    assert len(queue['jobs']) == 10
    for job in queue['jobs']:
        assert hashlib.sha256((ROOT / job['config']).read_bytes()).hexdigest() == job['config_sha256']
    report = compare(path, ROOT / 'configs/reproducibility/saits/references.v1.json')
    if report['pending_jobs']:
        for record in report['records']:
            if len(record['completed_seeds']) < 5:
                assert record['verdict'] == 'incomplete' and 'mean' not in record


def test_real_author_models_and_resume_are_verified_without_benchmark_claims():
    path = ROOT / 'experiments/runs/flow_matching_campaign/saits_cpu_preflight_report.json'
    if not path.exists():
        pytest.skip('Optional real-data integration preflight')
    report = json.loads(path.read_text(encoding='utf-8'))
    assert report['diagnostic'] and report['status'] == 'passed'
    assert {r['model'] for r in report['records']} == {'saits', 'brits'}
    assert all(all(r['checks'].values()) for r in report['records'])
