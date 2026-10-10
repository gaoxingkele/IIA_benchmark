"""Verify original Monash archives and byte identity with existing local raw data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

from scripts.data_acquisition.download_public_datasets import download_file, checksum_ok
from scripts.flow_matching.spectral_table2_bootstrap_v1 import sha, write

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    target=ROOT/config['output'];target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError('Preserve original acquisition evidence')
    records=[]
    for source in config['sources']:
        metadata=ROOT/source['metadata_path']
        if sha(metadata)!=source['metadata_sha256']:raise ValueError('Official record metadata differs')
        download_file(source)
        archive=ROOT/source['path']
        assert checksum_ok(archive,source['checksum'])
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:raise ValueError('Original archive CRC failed')
            member=source['filename'];raw=z.read(member)
        digest=hashlib.sha256(raw).hexdigest()
        prior=ROOT/source['local_raw_path']
        prior_bytes=prior.read_bytes()
        bitwise_identical=prior_bytes==raw
        normalized_text_identical=prior_bytes.replace(b'\r\n',b'\n')==raw.replace(b'\r\n',b'\n')
        if not normalized_text_identical:raise ValueError('Local raw content differs beyond line endings; review required')
        destination=ROOT/source['raw_path']
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():raise FileExistsError('Never overwrite downloaded raw data')
        with destination.open('xb') as f:f.write(raw)
        records.append(dict(dataset=source['id'],original_archive=source['path'],
            publisher_checksum=source['checksum'],archive_sha256=sha(archive),
            raw_path=source['raw_path'],raw_sha256=digest,raw_bytes=len(raw),
            existing_local_raw_path=source['local_raw_path'],existing_raw_bitwise_identical=bitwise_identical,
            line_ending_normalized_text_identical=normalized_text_identical,
            raw_input_used='Unmodified publisher archive member; prior local raw preserved',
            metadata_path=source['metadata_path'],metadata_sha256=sha(metadata),
            original_publisher_url=source['url'],archive_crc_passed=True))
    write(target,dict(passed=True,raw_files_preserved=True,records=records,
        config_sha256=sha(ROOT/args.config),captured_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(original_datasets_verified=len(records),publisher_raw_files_preserved=True)))


if __name__=='__main__':main()
