import copy
import hashlib
import json

import pytest

from scripts.flow_matching.recovery_results import resolve, sha, verified_attempts


def fixture(tmp_path):
    original={'id':'method__dataset__seed1','output_directory':'old/seed1','dataset':'dataset',
              'model_config':'model.json','seed':1,'frozen_files':[]}
    recovered=dict(original,id=original['id']+'__retry',output_directory='recovery/seed1')
    parent=tmp_path/'parent.json'
    parent.write_text(json.dumps({'jobs':[original],'evaluation':{'threshold':1}}))
    case={'original_id':original['id'],'original_job':original,'job':recovered,'kind':'strict',
          'original_queue':'parent.json','original_queue_sha256':sha(parent),
          'settings':{},'evaluation':{'threshold':1}}
    out=tmp_path/recovered['output_directory']
    out.mkdir(parents=True)
    (out/'model.pt').write_bytes(b'model fixture')
    (out/'scores.npz').write_bytes(b'score fixture')
    record={'status':'completed','experiment_sha256':hashlib.sha256(json.dumps(recovered,sort_keys=True).encode()).hexdigest(),
            'scores_sha256':sha(out/'scores.npz'),'checkpoint_sha256':sha(out/'model.pt')}
    (out/'result.json').write_text(json.dumps(record))
    queue=tmp_path/'recovery.json'
    queue.write_text(json.dumps({'jobs':[case]}))
    return original,case,queue,out


def test_recovery_resolves_same_original_slot_preserving_artifacts_and_seed_count(tmp_path):
    original,case,queue,out=fixture(tmp_path)
    before={p.name:p.read_bytes() for p in out.iterdir()}
    attempts=verified_attempts(tmp_path,['recovery.json'])
    chosen=resolve(original,attempts)
    assert chosen['recovery_job']==case['job'] and chosen['original_job']==original
    assert chosen['result_path']=='recovery/seed1/result.json'
    assert len([j for j in [original] if resolve(j,attempts)])==1
    assert {p.name:p.read_bytes() for p in out.iterdir()}==before


@pytest.mark.parametrize('change',['seed','evaluation','receipt','score'])
def test_changed_seed_protocol_identity_or_artifact_is_rejected(tmp_path,change):
    original,case,queue,out=fixture(tmp_path)
    if change=='seed':case['job']['seed']=2
    elif change=='evaluation':case['evaluation']['threshold']=5
    elif change=='receipt':
        result=json.loads((out/'result.json').read_text());result['experiment_sha256']='bad'
        (out/'result.json').write_text(json.dumps(result))
    elif change=='score':(out/'scores.npz').write_bytes(b'changed')
    queue.write_text(json.dumps({'jobs':[case]}))
    with pytest.raises(ValueError):verified_attempts(tmp_path,['recovery.json'])


def test_multiple_attempts_use_registered_order_without_metric_based_selection(tmp_path):
    original,case,queue,out=fixture(tmp_path)
    attempts=verified_attempts(tmp_path,['recovery.json'])
    second=copy.deepcopy(attempts[0]);second['result_path']='another/result.json'
    assert resolve(original,[attempts[0],second]) is attempts[0]
    assert resolve(original,[dict(attempts[0],status='not_completed'),second]) is second


def test_pending_attempt_never_promoted_to_completed(tmp_path):
    original,case,queue,out=fixture(tmp_path)
    (out/'result.json').unlink()
    attempts=verified_attempts(tmp_path,['recovery.json'])
    assert resolve(original,attempts) is None
