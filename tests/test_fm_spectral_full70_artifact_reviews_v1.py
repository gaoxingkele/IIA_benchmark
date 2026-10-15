from copy import deepcopy
import json
import subprocess
import sys

import pytest

from scripts.flow_matching import run_spectral_full70_artifact_reviews_v1 as s


def fixture():
    jobs = [dict(id=f'c{c}_s{seed}', model=f'm{c // 7}', dataset=f'd{c % 7}',
        data_track='registered', seed=seed) for c in range(14) for seed in range(42, 47)]
    catalog = dict(full_native_generators=30, records=[dict(canonical_id=j['id']) for j in jobs[:30]])
    config = dict(required_original_models=70, required_original_cohorts=14,
                  original_required_seeds=[42, 43, 44, 45, 46], preserved_reviewed_models=30)
    return config, dict(jobs=jobs), catalog


@pytest.mark.parametrize('fault', ['missing_job', 'duplicate_job', 'changed_seed', 'mixed_cohort',
    'seed_scope', 'missing_review', 'duplicate_review', 'foreign_review', 'reduced_obligation'])
def test_all_original_models_seeds_cohorts_and_preserved_reviews_are_mandatory(fault):
    c, q, r = fixture(); s.validate_scope(c, q, r)
    if fault == 'missing_job': q['jobs'].pop()
    elif fault == 'duplicate_job': q['jobs'][-1] = q['jobs'][-2]
    elif fault == 'changed_seed': q['jobs'][-1]['seed'] = 42
    elif fault == 'mixed_cohort': q['jobs'][-1]['data_track'] = 'different_distribution'
    elif fault == 'seed_scope': c['original_required_seeds'].pop()
    elif fault == 'missing_review': r['records'].pop()
    elif fault == 'duplicate_review': r['records'][-1] = r['records'][-2]
    elif fault == 'foreign_review': r['records'][-1]['canonical_id'] = 'unregistered'
    else: c['required_original_models'] = 30
    with pytest.raises(ValueError): s.validate_scope(c, q, r)


@pytest.mark.parametrize('field', ['id', 'seed', 'training_updates', 'data_shape', 'generation_batch_size', 'model_config'])
def test_fmri_recovery_changes_only_paths_not_training_or_sampling(field):
    job = dict(id='fmri43', seed=43, training_updates=15000, data_shape=[9977, 24, 50],
        generation_batch_size=2001, model_config='full.json', output_directory='old', artifact_directory='old_artifact')
    new = dict(job, output_directory='new', artifact_directory='new_artifact')
    s.recovery_binding(job, new)
    new[field] = 'changed'
    with pytest.raises(ValueError): s.recovery_binding(job, new)


def test_complete_source_waits_for_actual_guard_and_rejects_partial_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(s, 'ROOT', tmp_path)
    job = dict(id='j', output_directory='job', training_updates=12000, gradient_accumulate_every=2)
    (tmp_path / 'q.json').write_text(json.dumps(dict(state_root='state')))
    case = dict(job=job, queue='q.json')
    assert s.source_ready(case) is None
    (tmp_path / 'job').mkdir()
    result = dict(status='completed', job=job, benchmark_smoke=False, optimizer_updates=12000,
                  training_minibatches=24000, evaluation_repeats=5)
    (tmp_path / 'job/result.json').write_text(json.dumps(result))
    assert s.source_ready(case) is None
    (tmp_path / 'state/receipts').mkdir(parents=True)
    receipt = tmp_path / 'state/receipts/j.json'
    receipt.write_text(json.dumps(dict(exitcode=0, resource_guard_aborted=True, full_result_validated=True)))
    assert s.source_ready(case) is None
    receipt.write_text(json.dumps(dict(exitcode=0, resource_guard_aborted=False, full_result_validated=True)))
    assert s.source_ready(case)[2] == result
    result['training_minibatches'] = 23999
    (tmp_path / 'job/result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError): s.source_ready(case)


def test_registered_watch_includes_full_retry_and_remains_cold():
    code = ('from scripts.flow_matching import run_spectral_full70_artifact_reviews_v1 as s;'
        'import sys,json;c=s.registration();cases,old=s.verify(c);'
        'assert not any(k in sys.modules for k in ["numpy","scipy","torch"]);'
        'j=cases["spectral_original__diffusion_ts__fmri__released_author_data__seed43"];'
        'assert "actual500_full_retry" in j["queue"];'
        'print(json.dumps([len(cases),len(old["records"]),c["citation"]]))')
    p = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-c', code], cwd=s.ROOT,
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    assert json.loads(p.stdout) == [70, 30, 'https://arxiv.org/abs/2510.15366']


@pytest.mark.parametrize('field', ['settings', 'resource_lock', 'required_original_models', 'required_original_cohorts'])
def test_registration_cannot_weaken_scope_or_resources(field):
    c = deepcopy(s.registration())
    if field == 'settings': c[field]['minimum_commit_headroom_bytes'] = 1
    elif field == 'resource_lock': c[field] = 'another.lock'
    else: c[field] = 1
    with pytest.raises(ValueError): s.verify(c)
