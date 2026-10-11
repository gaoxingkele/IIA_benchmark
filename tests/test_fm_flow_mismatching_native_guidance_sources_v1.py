"""Native code identity/archive guards, not scientific performance."""
import subprocess
import zipfile

import pytest

from scripts.flow_matching.acquire_flow_mismatching_native_guidance_sources_v1 import extract_verified_author,verify_pdf


def original_fixture(tmp_path):
    git=tmp_path/'git';git.mkdir()
    subprocess.run(['git','init','-q',str(git)],check=True)
    (git/'method.py').write_bytes(b'def original():\n    return 1\n')
    (git/'weights.bin').write_bytes(bytes(range(12)))
    subprocess.run(['git','-C',str(git),'add','.'],check=True)
    subprocess.run(['git','-C',str(git),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','original'],check=True)
    commit=subprocess.check_output(['git','-C',str(git),'rev-parse','HEAD']).decode().strip()
    archive=tmp_path/'original.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for name in ['method.py','weights.bin']:z.writestr('FlowGrad-'+commit+'/'+name,(git/name).read_bytes())
    return git,commit,archive


def test_complete_original_archive_matches_every_git_blob_and_bytes(tmp_path):
    git,commit,archive=original_fixture(tmp_path)
    target=tmp_path/'complete'
    rows=extract_verified_author(archive,target,git,commit)
    assert len(rows)==2 and all(len(r['verified_git_blob'])==40 for r in rows)
    assert (target/'weights.bin').read_bytes()==bytes(range(12))
    with pytest.raises(FileExistsError):extract_verified_author(archive,target,git,commit)


def test_modified_author_bytes_cannot_pass_crc_as_original_code(tmp_path):
    git,commit,archive=original_fixture(tmp_path)
    forged=tmp_path/'forged.zip'
    with zipfile.ZipFile(forged,'w') as z:
        z.writestr('FlowGrad-'+commit+'/method.py',b'changed\n')
        z.writestr('FlowGrad-'+commit+'/weights.bin',bytes(range(12)))
    with pytest.raises(ValueError,match='blob byte mismatch'):
        extract_verified_author(forged,tmp_path/'forged_out',git,commit)


def test_incomplete_original_archive_is_not_full_code(tmp_path):
    git,commit,archive=original_fixture(tmp_path)
    subset=tmp_path/'subset.zip'
    with zipfile.ZipFile(subset,'w') as z:z.writestr('FlowGrad-'+commit+'/method.py',(git/'method.py').read_bytes())
    with pytest.raises(ValueError,match='Every original'):
        extract_verified_author(subset,tmp_path/'subset_out',git,commit)


def test_html_and_empty_pdf_do_not_count_as_original_full_text(tmp_path):
    page=tmp_path/'error.pdf';page.write_bytes(b'<html>Access required</html>')
    with pytest.raises(ValueError,match='not a PDF'):verify_pdf(page)
    from pypdf import PdfWriter
    writer=PdfWriter();writer.add_blank_page(width=100,height=100)
    with page.open('wb') as f:writer.write(f)
    with pytest.raises(ValueError,match='nonempty'):verify_pdf(page)
