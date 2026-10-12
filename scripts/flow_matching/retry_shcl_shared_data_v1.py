"""Retry five error payloads in a fresh directory; preserve all raw predecessors."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def inspect_csv(path, specification):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as stream:
        rows = csv.reader(stream); header = next(rows)
        if len(header) not in specification['allowed_column_counts']:
            raise ValueError('Invalid CSV header, including server/HTML error payload')
        count = 0
        for row in rows:
            if len(row) != len(header):
                raise ValueError('Malformed or incomplete CSV row')
            count += 1
        if count != specification['expected_rows']:
            raise ValueError('Full original expected row count required: ' + str(count))
    return dict(full_rows=count, columns=len(header), header=header,
        complete_csv_structure_verified=True, numerical_loader_and_label_equivalence_pending=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    args = parser.parse_args(); config = read(ROOT / args.config)
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen acquisition binding changed: ' + pin['path'])
    output = Path(config['download_root'])
    if output.exists():
        raise FileExistsError('Every registered download attempt must be preserved')
    output.mkdir(parents=True)
    records = []
    for source in config['sources']:
        old = ROOT / source['preserved_previous_file']
        if sha(old) != source['preserved_previous_sha256']:
            raise ValueError('Original failed payload changed before retry')
        directory = output / source['dataset']; directory.mkdir(exist_ok=True)
        command = [config['aria2c'], '--all-proxy=' + config['proxy'], '--max-tries=2',
            '--retry-wait=3', '--connect-timeout=15', '--timeout=30',
            '--file-allocation=none', '--auto-file-renaming=false', '--allow-overwrite=false',
            '--max-connection-per-server=2', '--split=2', '--console-log-level=warn',
            '--summary-interval=0', '--dir=' + str(directory), '--out=' + source['filename'], source['url']]
        log = output / (source['dataset'] + '_' + source['filename'] + '.download.log')
        with log.open('xb') as stream:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            code = process.wait()
        path = directory / source['filename']
        record = dict(source=source, exit_code=code, actual_command=command,
            downloaded_file_present=path.is_file(), passed=False, previous_failed_payload_preserved=True)
        if path.is_file():
            record.update(bytes=path.stat().st_size, sha256=sha(path))
            try:
                if code != 0: raise ValueError('Download exit code was nonzero')
                record['full_csv_audit'] = inspect_csv(path, source)
                # Independent reread detects a changed file, not a publisher checksum.
                if sha(path) != record['sha256']:
                    raise ValueError('Downloaded file changed during complete CSV audit')
                record['passed'] = True
            except (ValueError, UnicodeError, StopIteration, csv.Error) as error:
                record['failure'] = str(error)
        if sha(old) != source['preserved_previous_sha256']:
            raise ValueError('Original raw failure payload changed during retry')
        records.append(record)
        exclusive(output / ('receipt_' + source['dataset'] + '_' + source['filename'] + '.json'), record)
        print(json.dumps(dict(dataset=source['dataset'], file=source['filename'], passed=record['passed'],
            bytes=record.get('bytes'), exit_code=code)), flush=True)
    report = dict(captured_utc=datetime.now(timezone.utc).isoformat(), configuration=args.config,
        configuration_sha256=sha(ROOT / args.config), expected_files=5,
        complete_valid_csv_files=sum(row['passed'] for row in records), records=records,
        preserved_raw_files_unchanged=True, publisher_checksum_available=False,
        local_SHA256_reread_verified=True, all_original_experiments_complete=False)
    exclusive(output / 'acquisition_report.json', report)


if __name__ == '__main__':
    main()
