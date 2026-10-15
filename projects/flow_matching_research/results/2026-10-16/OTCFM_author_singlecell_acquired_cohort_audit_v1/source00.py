"""Acquire publisher-checksummed author single-cell assets without overwrites."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from scripts.data_acquisition.download_public_datasets import find_aria2
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha

CONFIG = 'configs/acquisition/fm_otcfm_author_singlecell_mendeley.v1.json'
API = 'https://data.mendeley.com/public-api/datasets/hhny5ff7yj/files?folder_id=root&version=1'
BASE = 'data/public_datasets/flow_matching/project_additions/otcfm_author_singlecell_mendeley'
FILES = {'ebdata_v3.h5ad', 'op_cite_inputs_0.h5ad', 'op_train_multi_targets_0.h5ad'}
NOTEBOOK = 'experiments/runs/flow_matching_campaign/sources/ot_cfm/original/examples/single_cell/single-cell_example.ipynb'


def exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def sources(payload):
    if not isinstance(payload, list) or {r['filename'] for r in payload} != FILES or len(payload) != 3:
        raise ValueError('Exactly three public author version1 assets required')
    result = []
    for row in payload:
        details = row['content_details']
        if (row['status'] != 'COMPLETED' or details['size'] != row['size'] or row['size'] <= 0
                or not re.fullmatch('[0-9a-f]{64}', details['sha256_hash'])
                or not re.fullmatch('[0-9a-f-]{36}', row['id'])):
            raise ValueError('Complete publisher size, SHA256 and file identity required')
        url = 'https://data.mendeley.com/public-files/datasets/hhny5ff7yj/files/'+row['id']+'/file_downloaded'
        if details['download_url'] != url:
            raise ValueError('Exact public publisher content URL required')
        result.append(dict(id='otcfm_author_'+row['filename'].removesuffix('.h5ad'),
            filename=row['filename'], path=BASE+'/'+row['filename'], url=url,
            expected_bytes=row['size'], checksum='sha256:'+details['sha256_hash'],
            file_id=row['id'], content_id=details['id'], format='hdf5', license='CC BY 4.0',
            citation='https://doi.org/10.17632/hhny5ff7yj.1', author_paper='https://arxiv.org/abs/2302.00482',
            boundary='Direct author-published processed artifact; inspect actual donor, days, PC arrays and '
                     'preprocessing before paper-equivalence or numerical benchmark claims.'))
    return sorted(result, key=lambda item: item['expected_bytes'])


def validate(path, item):
    path = Path(path)
    if path.stat().st_size != item['expected_bytes']:
        raise ValueError('Exact publisher payload size required')
    with path.open('rb') as stream:
        if stream.read(8) != b'\x89HDF\r\n\x1a\n':
            raise ValueError('A real HDF5 payload is required')
    digest = sha(path)
    if item['checksum'] != 'sha256:'+digest:
        raise ValueError('Publisher SHA256 mismatch; preserve original attempt')
    return dict(bytes=path.stat().st_size, sha256=digest, publisher_checksum_verified=True,
                HDF5_signature_verified=True)


def acquire(item, config):
    target = ROOT/item['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return dict(id=item['id'], status='existing_publisher_verified', **validate(target, item))
    partial = target.with_name(target.name+'.partial')
    aria2 = find_aria2()
    if aria2 is None:
        raise RuntimeError('Registered preferred aria2c executable is required')
    command = [aria2, '--all-proxy='+config['proxy'], '--continue=true', '--check-integrity=true',
        '--checksum=sha-256='+item['checksum'].split(':')[1], '--allow-overwrite=false',
        '--auto-file-renaming=false', '--file-allocation=none', '--split=8',
        '--max-connection-per-server=8', '--min-split-size=8M', '--max-tries=3', '--retry-wait=5',
        '--connect-timeout=25', '--timeout=60', '--console-log-level=notice', '--summary-interval=15',
        '--dir='+str(target.parent.resolve()), '--out='+partial.name, item['url']]
    print(json.dumps(dict(id=item['id'], state='downloading_registered_publisher_payload', bytes=item['expected_bytes'])),flush=True)
    attempt = 1
    while any(target.with_name(target.name+'.attempt%03d.%s.log'%(attempt,kind)).exists() for kind in ['stdout','stderr']):
        attempt += 1
    with target.with_name(target.name+'.attempt%03d.stdout.log'%attempt).open('xb') as out, target.with_name(target.name+'.attempt%03d.stderr.log'%attempt).open('xb') as err:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        code = process.wait()
    if code != 0:
        raise RuntimeError('aria2c failed with exit '+str(code)+'; partial payload/logs preserved')
    proof = validate(partial, item)
    if target.exists():
        raise FileExistsError('Preserve preexisting raw author payload')
    # Windows rename refuses to replace a destination that appeared meanwhile.
    partial.rename(target)
    receipt = dict(id=item['id'], path=item['path'], resolved_path=str(target.resolve()),
        status='downloaded_publisher_size_SHA256_and_HDF5_verified', url=item['url'],
        backend='aria2c', proxy=config['proxy'], command=command, returncode=code,
        captured_utc=datetime.now(timezone.utc).isoformat(), **proof)
    exclusive(target.with_name(target.name+'.verified.json'), receipt)
    print(json.dumps(dict(id=item['id'], state='publisher_checksum_verified', bytes=proof['bytes'])),flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.register:
        from curl_cffi import requests
        response = requests.get(API, proxy='http://127.0.0.1:17890', impersonate='chrome', timeout=30)
        response.raise_for_status()
        payload = response.json()
        items = sources(payload)
        source = ROOT/BASE/'publisher_version1_files.json'
        exclusive(source, payload)
        exclusive(ROOT/CONFIG, dict(schema_version=1, proxy='http://127.0.0.1:17890',
            preferred_download_tool='aria2c', maximum_parallel_downloads=2, dataset_doi='10.17632/hhny5ff7yj.1',
            citation='https://data.mendeley.com/datasets/hhny5ff7yj/1', license='CC BY 4.0',
            publisher_metadata_path=str(source.relative_to(ROOT)), publisher_metadata_sha256=sha(source),
            evidence_notebook=NOTEBOOK, evidence_notebook_sha256=sha(ROOT/NOTEBOOK),
            source_receipts=[dict(path=str(Path(__file__).relative_to(ROOT)), sha256=sha(Path(__file__)))],
            sources=items, all_original_experiments_complete=False))
        print(json.dumps(dict(registered=True, payloads=len(items), total_bytes=sum(i['expected_bytes'] for i in items))))
        return
    if not args.download:
        parser.error('Choose --register or --download')
    config = read(ROOT/CONFIG)
    assert sha(ROOT/config['publisher_metadata_path']) == config['publisher_metadata_sha256']
    assert sources(read(ROOT/config['publisher_metadata_path'])) == config['sources']
    assert sha(ROOT/config['evidence_notebook']) == config['evidence_notebook_sha256']
    for pin in config['source_receipts']:
        assert sha(ROOT/pin['path']) == pin['sha256']
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    owner = resource_api()['exclusive_lock'](ROOT/BASE/'download.lock')
    if owner is None:
        raise RuntimeError('Existing author-data download process preserved')
    try:
        results = []
        with ThreadPoolExecutor(max_workers=config['maximum_parallel_downloads']) as executor:
            futures=[executor.submit(acquire,item,config) for item in config['sources']]
            for future in as_completed(futures):
                results.append(future.result())
        exclusive(ROOT/BASE/'all_author_payloads_verified.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
            configuration=CONFIG, configuration_sha256=sha(ROOT/CONFIG), records=results,
            verified_payloads=len(results), original_raw_files_overwritten=False,
            model_training_or_data_preprocessing_executed=False, all_original_experiments_complete=False))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
