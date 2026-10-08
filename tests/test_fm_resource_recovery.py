import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def test_recovery_preserves_every_original_algorithm_budget_dataset_and_seed():
    queue=json.loads((ROOT/'configs/experiments/fm_resource_recovery_queue.v2.json').read_text())
    assert len(queue['jobs'])==11
    assert {j['kind'] for j in queue['jobs']}=={'industrial','strict','pi','maelnet'}
    for case in queue['jobs']:
        new,old=case['job'],case['original_job']
        changed={key for key in new if new[key]!=old[key]}
        assert changed=={'id','output_directory'}
        assert new['seed']==old['seed'] and new['dataset']==old['dataset']
        assert new['output_directory'].startswith(queue['settings']['output_root'])
        assert new['output_directory']!=old['output_directory']
        assert case['evidence'] and case['original_queue_sha256']
    assert queue['settings']['emergency_commit_headroom_bytes']==8*1024**3


def test_every_original_binding_is_an_existing_registered_job():
    queue=json.loads((ROOT/'configs/experiments/fm_resource_recovery_queue.v2.json').read_text())
    for case in queue['jobs']:
        original=json.loads((ROOT/case['original_queue']).read_text())
        assert case['original_job']==next(j for j in original['jobs'] if j['id']==case['original_id'])
