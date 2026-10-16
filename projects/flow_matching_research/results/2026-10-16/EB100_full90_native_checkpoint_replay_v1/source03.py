import ast
from pathlib import Path
import pytest
from scripts.flow_matching import audit_otcfm_EB100_full90_replay_v1 as s


def test_every_complete_original_timepoint_population_is_kept():
    assert s.interior_geometry([2381,4163,3278,3665,3332])==[(2381,4163),(4163,3278),(3278,3665)]


@pytest.mark.parametrize('sizes',[[10,20,30,40],[10,0,30,40,50],[10,20,-1,40,50],[10,20,30,40,50,60]])
def test_original_EB100_replay_cannot_silently_reuse_CITE_or_truncated_geometry(sizes):
    with pytest.raises(ValueError):s.interior_geometry(sizes)


def test_all90_original_inputs_and_both_existing_replay_queues_remain_required():
    cfg=s.registration();q=s.read(s.ROOT/cfg['original_queue'])
    assert cfg['required_full_native_cases']==len(cfg['required_case_ids'])==len(set(cfg['required_case_ids']))==90
    assert cfg['required_case_ids']==[j['id'] for j in q['jobs']]
    assert cfg['settings']==q['settings'] and cfg['native_metric_tolerance']==dict(rtol=2e-6,atol=2e-6)
    assert cfg['EB100_complete_metric_count']==44
    assert cfg['resource_lock']=='experiments/runs/fm_otcfm_singlecell_readonly_replay_cpu.lock'
    assert cfg['independent_source_replay_queue_preserved']=='configs/reproducibility/fm_otcfm_singlecell_full120_checkpoint_replay.v2.json'
    assert cfg['preserved_full150_replay_queue']=='configs/reproducibility/fm_otcfm_EB_CITE_full150_checkpoint_replay.v1.json'
    ids=cfg['required_case_ids'][:]
    for key in ['independent_source_replay_queue_preserved','preserved_full150_replay_queue']:
        ids+=s.read(s.ROOT/cfg[key])['required_case_ids']
    assert len(ids)==len(set(ids))==360


def test_entire_checkpoint_reinference_and_full_independent_LP_worker_remains_identical():
    def body(path):
        tree=ast.parse((s.ROOT/path).read_text(encoding='utf-8'))
        return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker'),include_attributes=False)
    assert body(s.MODULE.replace('.','/')+'.py')==body('scripts/flow_matching/audit_otcfm_EB_CITE_full150_replay_v1.py')
