from copy import deepcopy
import hashlib
from pathlib import Path
import pytest
from scripts.flow_matching import acquire_otcfm_author_singlecell_mendeley_v1 as s


def payload():
    rows=[]
    for index,name in enumerate(sorted(s.FILES)):
        identity=str(index)*8+'-1111-2222-3333-'+'a'*12
        rows.append(dict(filename=name,id=identity,size=10,status='COMPLETED',content_details=dict(
            id='content'+str(index),size=10,sha256_hash='a'*64,
            download_url='https://data.mendeley.com/public-files/datasets/hhny5ff7yj/files/'+identity+'/file_downloaded')))
    return rows


@pytest.mark.parametrize('fault',['missing','duplicate','foreign_url','wrong_hash','unfinished','wrong_size'])
def test_untrusted_publisher_response_cannot_change_file_identity_or_integrity(fault):
    rows=payload()
    assert len(s.sources(rows))==3
    if fault=='missing': rows.pop()
    elif fault=='duplicate': rows[1]=deepcopy(rows[0])
    elif fault=='foreign_url': rows[0]['content_details']['download_url']='https://other.invalid/file'
    elif fault=='wrong_hash': rows[0]['content_details']['sha256_hash']='missing'
    elif fault=='unfinished': rows[0]['status']='PENDING'
    else: rows[0]['content_details']['size']=100
    with pytest.raises(ValueError): s.sources(rows)


@pytest.mark.parametrize('fault',['correct','wrong_size','wrong_checksum','html_payload'])
def test_full_size_signature_and_publisher_digest_required(tmp_path,fault):
    content=b'\x89HDF\r\n\x1a\n'+b'checksum-test-payload'
    path=tmp_path/'file.h5ad'
    if fault=='html_payload': content=b'<html>request failed</html>'
    path.write_bytes(content)
    item=dict(expected_bytes=len(content),checksum='sha256:'+hashlib.sha256(content).hexdigest())
    if fault=='wrong_size': item['expected_bytes']+=1
    if fault=='wrong_checksum': item['checksum']='sha256:'+'0'*64
    if fault=='correct': assert s.validate(path,item)['publisher_checksum_verified']
    else:
        with pytest.raises(ValueError): s.validate(path,item)


def test_existing_raw_file_is_preserved_without_starting_downloader(tmp_path,monkeypatch):
    path=tmp_path/'original.h5ad';content=b'original user bytes';path.write_bytes(content)
    monkeypatch.setattr(s,'ROOT',tmp_path)
    monkeypatch.setattr(s,'find_aria2',lambda:pytest.fail('No download may overwrite a preexisting raw file'))
    with pytest.raises(ValueError):
        s.acquire(dict(id='existing_raw',path=path.name,expected_bytes=99),{})
    assert path.read_bytes()==content
