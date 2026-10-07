"""Source identity, preservation and truthful evidence boundaries for FM ARA."""
import copy
import hashlib
import io
import json
from pathlib import Path
import zipfile

import fitz
import pytest

from scripts.literature.import_flow_matching_archives import import_archives
from scripts.literature.build_flow_matching_ara import inspect_source, validate_annotations


def bundle(tmp_path, name='source.pdf', paper_id='source'):
    with fitz.open() as doc:
        doc.new_page().insert_text((72,72),'Proceedings: unrelated chapter')
        doc.new_page().insert_text((72,72),'Registered Research Title: graph method')
        raw=doc.tobytes()
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as archive: archive.writestr(name,raw)
    payload=stream.getvalue();(tmp_path/'input.zip').write_bytes(payload)
    digest=hashlib.sha256(raw).hexdigest()
    config={'output_directory':'imported','archives':[{'path':'input.zip','sha256':hashlib.sha256(payload).hexdigest(),
            'members':[{'name':name,'paper_id':paper_id,'sha256':digest,
                        'title_tokens':['Registered Research Title'],'selected_pdf_pages':[2]}]}]}
    return config,raw


def test_selected_chapter_import_preserves_original_and_idempotence(tmp_path):
    config,raw=bundle(tmp_path); before=(tmp_path/'input.zip').read_bytes()
    report=import_archives(tmp_path,config)
    assert report['records'][0]['pages']==2
    assert report['records'][0]['selected_pdf_pages']==[2]
    assert (tmp_path/report['records'][0]['path']).read_bytes()==raw
    assert import_archives(tmp_path,config)==report
    assert (tmp_path/'input.zip').read_bytes()==before


@pytest.mark.parametrize('name',['../outside.pdf','C:/outside.pdf','folder/../../outside.pdf'])
def test_archive_path_escape_rejected(tmp_path,name):
    config,_=bundle(tmp_path,name=name)
    with pytest.raises(ValueError,match='Unsafe archive'):import_archives(tmp_path,config)
    assert not (tmp_path/'imported').exists()


def test_existing_corrupt_destination_is_preserved(tmp_path):
    config,_=bundle(tmp_path); member=config['archives'][0]['members'][0]
    target=tmp_path/'imported/source'/(member['sha256']+'.pdf')
    target.parent.mkdir(parents=True);target.write_bytes(b'user existing bytes')
    with pytest.raises(FileExistsError,match='preserved'):import_archives(tmp_path,config)
    assert target.read_bytes()==b'user existing bytes'


def test_title_and_identifier_validation_before_writing(tmp_path):
    config,_=bundle(tmp_path);member=config['archives'][0]['members'][0]
    member['selected_pdf_pages']=[1]
    with pytest.raises(ValueError,match='title'):import_archives(tmp_path,config)
    member['selected_pdf_pages']=[2];member['paper_id']='../../outside'
    with pytest.raises(ValueError,match='identifier'):import_archives(tmp_path,config)
    assert not (tmp_path/'imported').exists()


def test_archive_checksum_mismatch_rejected(tmp_path):
    config,_=bundle(tmp_path);config['archives'][0]['sha256']='0'*64
    with pytest.raises(ValueError,match='Archive SHA256'):import_archives(tmp_path,config)


def test_source_hash_and_chapter_range_gate(tmp_path):
    config,_=bundle(tmp_path);r=import_archives(tmp_path,config)['records'][0]
    pages=inspect_source(tmp_path,r)
    assert set(pages)=={2}
    invalid={**r,'selected_pdf_pages':[3]}
    with pytest.raises(ValueError,match='page selection'):inspect_source(tmp_path,invalid)
    invalid={**r,'sha256':'0'*64}
    with pytest.raises(ValueError,match='SHA256'):inspect_source(tmp_path,invalid)


def test_anchors_and_results_cannot_escape_edition_or_claim_reproduction():
    paper={'id':'source','primary_source_sha256':'a','scientific_reproduction_status':'not_established_by_this_artifact',
           'anchors':[{'id':'A01','pdf_page':2,'marker':'graph'}],
           'reported_results':[{'pdf_page':2,'status':'author_reported_not_locally_reproduced'}]}
    pages={'a':{2:'Registered title graph'}}
    validate_annotations(paper,pages)
    for change in ['page','marker','reproduction','result_status']:
        bad=copy.deepcopy(paper)
        if change=='page':bad['reported_results'][0]['pdf_page']=1
        if change=='marker':bad['anchors'][0]['marker']='invented mechanism'
        if change=='reproduction':bad['scientific_reproduction_status']='reproduced'
        if change=='result_status':bad['reported_results'][0]['status']='local_passed'
        with pytest.raises(ValueError):validate_annotations(bad,pages)


def test_registered_collection_has_required_named_methods_and_no_fabricated_scores():
    root=Path(__file__).resolve().parents[1]
    registry=json.loads((root/'configs/reproducibility/flow_matching_ara.v1.json').read_text(encoding='utf-8'))
    papers={p['id']:p for p in registry['papers']}
    assert {'flow_matching','rectified_flow','ot_cfm','cfm_ts','jfi','maelnet','shcl','kgl','pi_transformer','tab'} <= papers.keys()
    assert all(p['scientific_reproduction_status']=='not_established_by_this_artifact' for p in papers.values())
    assert {r['dataset'] for r in papers['maelnet']['reported_results']}=={'MSL','PSM'}
    assert papers['swat_dataset']['sources'][0]['selected_pdf_pages']==list(range(100,112))
