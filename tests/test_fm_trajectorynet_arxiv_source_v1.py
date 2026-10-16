import io
import tarfile
import pytest
from scripts.flow_matching import acquire_TrajectoryNet_arxiv_source_v1 as s


@pytest.mark.parametrize('name',['../outside.tex','/abs.tex','C:/abs.tex',r'a\..\out.tex'])
def test_external_paths_cannot_overwrite_raw_data_or_escape_new_source_directory(name):
    with pytest.raises(ValueError):s.safe_member(name)


def archive(path,entries):
    with tarfile.open(path,'w:gz') as out:
        for name,data in entries:
            info=tarfile.TarInfo(name);info.size=len(data);out.addfile(info,io.BytesIO(data))


def test_every_member_is_read_and_only_text_is_extracted_without_execution(tmp_path):
    path=tmp_path/'source.tar';archive(path,[('main.tex',b'not executed'),('figure.pdf',b'%PDF fixture')])
    limits=dict(maximum_compressed_bytes=10000,maximum_uncompressed_bytes=65536,maximum_members=10)
    rows=s.inspect_archive(path,tmp_path/'text',limits)
    assert len(rows)==2 and all(r['sha256'] for r in rows)
    assert (tmp_path/'text/main.tex').read_bytes()==b'not executed'
    assert not (tmp_path/'text/figure.pdf').exists()
    with pytest.raises(FileExistsError):s.inspect_archive(path,tmp_path/'text',limits)


def test_duplicate_members_and_expansion_excess_are_rejected(tmp_path):
    path=tmp_path/'duplicate.tar';archive(path,[('main.tex',b'a'),('main.tex',b'b')])
    limits=dict(maximum_compressed_bytes=10000,maximum_uncompressed_bytes=65536,maximum_members=10)
    with pytest.raises(ValueError,match='duplicate'):s.inspect_archive(path,tmp_path/'duplicate_text',limits)
    path=tmp_path/'oversized.tar';archive(path,[('main.tex',b'x'*3000)])
    limits['maximum_uncompressed_bytes']=100
    with pytest.raises(ValueError,match='Expanded'):s.inspect_archive(path,tmp_path/'too_large',limits)
