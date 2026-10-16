import ast
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import pytest
from scripts.flow_matching import run_otcfm_EB_CITE_native150_v1 as s


def test_all150_full_original_conditions_include_every_EB_timepoint_and_all_five_seeds():
    cases=s.scientific_cases()
    assert len(cases)==150 and len({j['id'] for j in cases})==150
    assert Counter(j['dataset'] for j in cases)=={'released_author_EB':90,'released_author_CITE':60}
    for dataset,times in [('released_author_EB',{1,2,3}),('released_author_CITE',{1,2})]:
        group=[j for j in cases if j['dataset']==dataset]
        assert {j['leaveout_timepoint'] for j in group}==times
        assert {j['seed'] for j in group}=={42,43,44,45,46}
        assert all(j['scientific_protocol']['max_epochs']==1000 and j['expected_model_parameters']==9093 for j in group)
    assert len(s.preflight_cases(dict(jobs=cases)))==30


@pytest.mark.parametrize('fault',['missing_EB_leave3','missing_method','changed_recipe','duplicate'])
def test_preflight_cannot_drop_an_actual_scientific_branch(fault):
    cases=deepcopy(s.scientific_cases())
    if fault=='missing_EB_leave3':cases=[j for j in cases if not (j['dataset']=='released_author_EB' and j['leaveout_timepoint']==3)]
    elif fault=='missing_method':cases=[j for j in cases if j['method']!='sbcfm']
    elif fault=='changed_recipe':cases[0]['protocol']='shortened_recipe'
    else:cases.append(deepcopy(cases[0]))
    with pytest.raises(ValueError):s.preflight_cases(dict(jobs=cases))


def test_only_update_accounting_changes_the_full_frozen_original_worker_body():
    root=Path(__file__).resolve().parents[1]
    def function(path):
        return next(n for n in ast.parse((root/path).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='worker')
    original=function('scripts/flow_matching/run_otcfm_singlecell_native_v4.py')
    extended=function('scripts/flow_matching/run_otcfm_EB_CITE_native150_v1.py')
    names=[n.targets[0].id for n in extended.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)]
    assert names.index('full_training_optimizer_updates')<names.index('test')
    class AccountingOnly(ast.NodeTransformer):
        def visit_Assign(self,node):
            if isinstance(node.targets[0],ast.Name) and node.targets[0].id=='full_training_optimizer_updates':return None
            return self.generic_visit(node)
        def visit_Call(self,node):
            node=self.generic_visit(node)
            node.keywords=[k for k in node.keywords if k.arg!='posttest_selected_checkpoint_optimizer_updates']
            for keyword in node.keywords:
                if keyword.arg=='optimizer_updates' and isinstance(keyword.value,ast.Name) and keyword.value.id=='full_training_optimizer_updates':
                    keyword.value=ast.parse('int(trainer.global_step)',mode='eval').body
            return node
    normalized=AccountingOnly().visit(extended)
    assert ast.dump(normalized,include_attributes=False)==ast.dump(original,include_attributes=False)


def test_all_method_configs_preserve_native_architecture_and_source_yaml():
    for method in s.contract.METHODS:
        config=json.loads((s.ROOT/f'configs/models/fm_otcfm_EB_CITE_native__{method}.v1.json').read_text())
        original=json.loads((s.ROOT/f'configs/models/fm_otcfm_singlecell_native__{method}.v4.json').read_text())
        assert config['parameters']==original['parameters']
        assert config['native_author_yaml_sha256']==s.sha(s.ROOT/config['native_author_yaml'])
        assert config['callable_implementation'].endswith('run_otcfm_EB_CITE_native150_v1:instantiate_case')
        assert config['dataset_and_protocol_binding']==s.QUEUE and not config['historical_paper_equivalence_certified']


def test_registration_keeps_full270_scope_and_every_existing_guard():
    q=s.registration();parent=s.read(s.ROOT/s.original.QUEUE)
    assert len(q['jobs'])+len(parent['jobs'])==270
    assert q['settings']==parent['settings'] and q['resource_lock']==parent['resource_lock']
    assert q['seeds']==[42,43,44,45,46] and q['required_models']==150
    assert [r['path'] for r in q['source_receipts']]==sorted({r['path'] for r in q['source_receipts']})
    assert {j['data']['cells'] for j in q['jobs']}=={18203,31240}
