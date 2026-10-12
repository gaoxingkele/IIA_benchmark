"""Author frontend and full script budgets; operator checks are smoke only."""
import ast
import copy

import pytest

from scripts.flow_matching import shcl_released_native_v1 as native


def config():
    return native.read(native.ROOT/'configs/models/fm_shcl_released_vae_native.v1.json')


def test_exactly_one_unused_parser_declaration_added_all_other_author_statements_unchanged():
    source=(native.verify(config())/'main.py').read_text(encoding='utf-8')
    original=ast.parse(source);repaired=native.corrected_main_ast(source)
    guard=next(n for n in repaired.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
        and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')
    declarations=[i for i,n in enumerate(guard.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and n.value.args and isinstance(n.value.args[0],ast.Constant) and n.value.args[0].value=='--threshold']
    assert len(declarations)==1
    guard.body.pop(declarations[0])
    assert ast.dump(original)==ast.dump(repaired)


def test_actual_unrecognized_threshold_error_reproduced_and_corrected_native_parser_accepts_same_command():
    argv=['--mode','test','--dataset','SMAP','--threshold','0.02170261','--num_epochs','1','--win_size','110']
    with pytest.raises(SystemExit) as error:native.parse_author_cli(config(),argv,corrected=False)
    assert error.value.code==2
    repaired=native.parse_author_cli(config(),argv)
    assert repaired.threshold==.02170261 and repaired.num_epochs==1 and repaired.win_size==110


def test_all_provided_train_test_commands_and_entire_SMD_grid_preserved():
    commands=native.released_script_commands(config())
    assert len(commands)==413
    assert sum(c['mode']=='train' for c in commands)==49
    assert sum(c['mode']=='test' for c in commands)==364
    smd=[c for c in commands if c['script']=='SMD.sh']
    assert len(smd)==405
    assert {c['win_size'] for c in smd}==set(range(50,131,10))
    assert {c['num_epochs'] for c in smd}==set(range(1,6))
    assert {c['seed'] for c in commands}=={1}
    assert {c['anormly_ratio'] for c in smd if c['mode']=='test'}=={i/20 for i in range(1,9)}


def test_native_solver_does_not_consume_script_threshold_or_constructor_ignored_flags():
    source=native.verify(config())
    solver=ast.parse((source/'solver.py').read_text(encoding='utf-8'))
    assert not any(isinstance(n,ast.Attribute) and n.attr=='threshold' for n in ast.walk(solver))
    model=ast.parse((source/'model/SHCLVae.py').read_text(encoding='utf-8'))
    constructor=next(n for c in model.body if isinstance(c,ast.ClassDef) and c.name=='SHCLVae'
        for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    for flag in ('e_layers','fr','tr','c_out'):
        assert not any(isinstance(n,ast.Name) and n.id==flag for body in constructor.body for n in ast.walk(body))


def test_eval_mode_still_samples_original_latent_noise_operator_contract_is_not_performance():
    source=native.verify(config())
    tree=ast.parse((source/'model/SHCLVae.py').read_text(encoding='utf-8'))
    reparam=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='VAEEncoder'
        for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='reparameterize')
    assert any(isinstance(n,ast.Attribute) and n.attr=='randn_like' for n in ast.walk(reparam))
    assert not any(isinstance(n,ast.Attribute) and n.attr=='training' for n in ast.walk(reparam))
    assert 'original_environment_data_and_full_training_equivalence_pending' in config()['reproduction_status']
