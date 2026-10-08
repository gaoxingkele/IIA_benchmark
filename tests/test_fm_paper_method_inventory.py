import copy
import hashlib
import json

import pytest

from scripts.flow_matching.verify_paper_method_inventory import verify
from scripts.flow_matching.refresh_paper_method_inventory import matching_candidates


def fixture(tmp_path):
    data=b'preserved primary'; (tmp_path/'paper.pdf').write_bytes(data)
    sha=hashlib.sha256(data).hexdigest()
    cache=tmp_path/'cache/paper'; cache.mkdir(parents=True)
    (cache/(sha+'.json')).write_text(json.dumps({'source_sha256':sha,'pages':{'1':'Main Method; BaselineX'}}))
    (tmp_path/'method.py').write_text('raise RuntimeError("must not execute")')
    ara={'local_text_cache':'cache','papers':[{'id':'paper','primary_source_sha256':sha,
            'sources':[{'path':'paper.pdf','sha256':sha}]}]}
    inventory={'all_methods_complete':False,'all_ablations_complete':False,'all_experiment_tables_reviewed':False,
        'papers':[{'paper_id':'paper'}], 'record_counts_by_role':{'main':1},
        'records':[{'record_id':'paper:main','paper_id':'paper','primary_source_sha256':sha,
            'pdf_page':1,'marker':'Main Method','role':'main','matched_code_paths':['method.py'],'model_configs':[]}]}
    return inventory,ara


def test_primary_anchor_validation_never_executes_candidate(tmp_path):
    inventory,ara=fixture(tmp_path)
    assert verify(tmp_path,inventory,ara)['status']=='passed'


@pytest.mark.parametrize('error',['anchor','edition','duplicate','missing','claim','pdf_drift'])
def test_rejects_drift_and_unsupported_completion(tmp_path,error):
    inventory,ara=fixture(tmp_path); inventory=copy.deepcopy(inventory)
    record=inventory['records'][0]
    if error=='anchor':record['pdf_page']=2
    if error=='edition':record['primary_source_sha256']='wrong'
    if error=='duplicate':inventory['records'].append(copy.deepcopy(record))
    if error=='missing':record['matched_code_paths']=['absent.py']
    if error=='claim':inventory['all_methods_complete']=True
    if error=='pdf_drift':(tmp_path/'paper.pdf').write_bytes(b'changed')
    with pytest.raises(ValueError):verify(tmp_path,inventory,ara)


def test_alias_binding_respects_paper_identity_and_component_scope():
    source={'aliases':['OT-Flow'],'excluded_paper_bindings':['ambiguous_paper'],
            'related_component_aliases':['RealNVP'],
            'scoped_aliases':[{'alias':'LSTM-VAE','paper_ids':['shcl']}]}
    assert not matching_candidates({'method':'OT-Flow','paper_id':'ambiguous_paper'},source)
    assert matching_candidates({'method':'RealNVP','paper_id':'other'},source)==['maintainer_component_candidate']
    assert matching_candidates({'method':'LSTM-VAE','paper_id':'shcl'},source)==['reference_identity_reviewed_source_candidate']
    assert not matching_candidates({'method':'LSTM-VAE','paper_id':'other'},source)
    assert not matching_candidates({'method':'DSBM','paper_id':'shcl'},source)
