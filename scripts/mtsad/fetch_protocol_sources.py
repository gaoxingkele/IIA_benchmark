"""Download registered official source snapshots into the anomaly-protocol audit area."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching import fetch_author_sources as acquisition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/reproducibility/mtsad_protocol_sources.v1.json')
    arguments = parser.parse_args()
    configuration = json.loads(arguments.config.read_text(encoding='utf-8'))
    acquisition.BASE = ROOT / configuration['output_root']
    for paper in configuration['papers']:
        snapshot = acquisition.fetch(paper['id'], paper['author_repository'], configuration['proxy'], paper['author_commit'])
        expected = paper.get('archive_sha256')
        if expected and snapshot['archive_sha256'] != expected:
            raise ValueError('Source archive checksum mismatch: ' + paper['id'])
        print(json.dumps({'id': paper['id'], 'commit': snapshot['commit'], 'archive_sha256': snapshot['archive_sha256']}), flush=True)
    pdf_records = []
    from pypdf import PdfReader
    for paper in configuration.get('pdfs', []):
        target = ROOT / paper['target']
        if target.exists():
            payload = target.read_bytes()
        else:
            response = acquisition.requests.get(paper['url'], proxy=configuration['proxy'], impersonate='chrome', timeout=90)
            response.raise_for_status()
            payload = response.content
        if not payload.startswith(b'%PDF-'):
            raise ValueError('Downloaded HTML instead of a PDF: ' + paper['id'])
        digest = hashlib.sha256(payload).hexdigest()
        if paper.get('sha256') and digest != paper['sha256']:
            raise ValueError('PDF checksum mismatch: ' + paper['id'])
        document = PdfReader(io.BytesIO(payload), strict=True)
        first_page = ' '.join((document.pages[0].extract_text() or '').split())
        if paper['title'].casefold() not in first_page.casefold():
            raise ValueError('PDF identity does not match registered title')
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as output:
                output.write(payload)
        record = dict(paper, sha256=digest, bytes=len(payload), pages=len(document.pages), status='verified_pdf',
                      checksum_boundary='Locally computed SHA256; publisher checksum not supplied. Version pinned and first-page title verified.')
        pdf_records.append(record)
        print(json.dumps({'id': record['id'], 'sha256': record['sha256'], 'pages': record['pages']}), flush=True)
    (acquisition.BASE / 'pdf_download_manifest.json').write_text(json.dumps(pdf_records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
