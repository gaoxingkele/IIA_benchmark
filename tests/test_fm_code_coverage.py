"""Protect code-presence audit boundaries without executing vendor sources."""
import copy
import hashlib
import json

import pytest

from scripts.flow_matching.audit_code_coverage import (
    inspect_resource, inspect_entrypoint, inspect_variant_locator, paper_navigation, validate_config,
)


def test_resource_scan_does_not_execute_and_fingerprints_changes(tmp_path):
    base=tmp_path/'source';base.mkdir()
    module=base/'method.py'
    module.write_text('raise RuntimeError("must never run")\nclass Model: pass\n')
    (base/'ablation.png').write_bytes(b'figure is not source code')
    (base/'Ablation_Window.sh').write_text('echo run_variant\n')
    ignored=base/'__MACOSX';ignored.mkdir()
    (ignored/'._method.py').write_text('broken syntax !')
    report,files=inspect_resource(tmp_path,{'id':'method','path':'source'})
    assert report['python_files']==1
    assert report['python_syntax_issues_current_interpreter']==[]
    assert report['ablation_filename_candidates']==['Ablation_Window.sh']
    assert all('png' not in r['path'] and '__MACOSX' not in r['path'] for r in files)
    module.write_text('class Changed: pass\n')
    newer,_=inspect_resource(tmp_path,{'id':'method','path':'source'})
    assert report['code_tree_sha256']!=newer['code_tree_sha256']


def test_legacy_syntax_and_warnings_are_recorded(tmp_path):
    (tmp_path/'old.py').write_text('print "legacy"\n')
    (tmp_path/'escape.py').write_text('value = "\\q"\n')
    report,_=inspect_resource(tmp_path,{'id':'method','path':'.'})
    assert report['python_syntax_issues_current_interpreter'][0]['path']=='old.py'
    assert report['python_syntax_warnings_current_interpreter'][0]['path']=='escape.py'


def test_entrypoint_symbols_without_import_or_nested_false_positive(tmp_path):
    path=tmp_path/'src/iia_benchmark/models/method.py';path.parent.mkdir(parents=True)
    path.write_text('raise RuntimeError("no execution")\nclass Model: pass\ndef outer():\n    def nested(): pass\n')
    assert inspect_entrypoint(tmp_path,'iia_benchmark.models.method:Model')['status']=='top_level_symbol_found'
    assert inspect_entrypoint(tmp_path,'iia_benchmark.models.method:nested')['status']=='symbol_not_found'
    assert inspect_entrypoint(tmp_path,'scripts.missing:run')['status']=='missing_module'


def test_notebook_variant_locator_ignores_markdown_claims(tmp_path):
    path=tmp_path/'tutorial.ipynb'
    path.write_text(json.dumps({'cells':[{'cell_type':'markdown','source':['score_loss']},
                                       {'cell_type':'code','source':['flow_loss = 0']}]}))
    locator={'path':'tutorial.ipynb','markers':['score_loss','flow_loss']}
    assert inspect_variant_locator(tmp_path,locator)['status']=='selected_markers_missing'
    assert inspect_variant_locator(tmp_path,{**locator,'markers':['flow_loss']})['status']=='selected_code_markers_present'


@pytest.mark.parametrize('failure',['duplicate','unknown_paper','unknown_source','coverage','claim'])
def test_registry_rejects_ambiguous_coverage_and_completeness_claims(failure):
    config={'source_resources':[{'id':'source'}],'main_status_overrides':{'paper':{}},
            'paper_source_bindings':{'paper':['source']},'all_methods_complete':False,'all_ablations_complete':False}
    bad=copy.deepcopy(config)
    if failure=='duplicate':bad['source_resources'].append({'id':'source'})
    if failure=='unknown_paper':bad['paper_source_bindings']['invented']=[]
    if failure=='unknown_source':bad['paper_source_bindings']['paper']=['invented']
    if failure=='coverage':bad['main_status_overrides']={}
    if failure=='claim':bad['all_ablations_complete']=True
    with pytest.raises(ValueError):validate_config(bad,{'papers':[{'id':'paper'}]})


def test_ablation_navigation_rejects_wrong_edition_anchor_and_pdf_drift(tmp_path):
    source=tmp_path/'paper.pdf';source.write_bytes(b'preserved source')
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    paper={'id':'paper','primary_source_sha256':sha,'sources':[{'path':'paper.pdf','sha256':sha}]}
    cache=tmp_path/'cache/paper'/f'{sha}.json';cache.parent.mkdir(parents=True)
    content={'source_sha256':sha,'pages':{'1':'Ablation without graph'}}
    cache.write_text(json.dumps(content))
    ara={'local_text_cache':'cache','papers':[paper]}
    axes=[{'pdf_page':1,'marker':'without graph','axes':['graph off']}]
    assert paper_navigation(tmp_path,ara,paper,axes)['reviewed_axes']==axes
    with pytest.raises(ValueError,match='anchor'):paper_navigation(tmp_path,ara,paper,[{**axes[0],'pdf_page':2}])
    content['source_sha256']='wrong edition';cache.write_text(json.dumps(content))
    with pytest.raises(ValueError,match='edition'):paper_navigation(tmp_path,ara,paper,axes)
    content['source_sha256']=sha;cache.write_text(json.dumps(content));source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA256'):paper_navigation(tmp_path,ara,paper,axes)
