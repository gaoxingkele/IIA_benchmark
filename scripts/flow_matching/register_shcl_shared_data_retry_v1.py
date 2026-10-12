"""Bind SHCL's exact README Drive folder and fresh five-file retry destinations."""
import json
from pathlib import Path

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    catalog_path = 'configs/acquisition/flow_matching_drive_sources.json'
    catalog = read(ROOT / catalog_path)
    sources = []
    for original in catalog['sources']:
        if 'dcdetector/author_bundle/' not in original['path']: continue
        dataset = Path(original['path']).parts[-2]
        if dataset not in ('PSM', 'SWAT'): continue
        path = ROOT / original['path']
        if path.read_bytes() != b'Internal Server Error':
            raise ValueError('Only exact identified failed original payloads registered')
        filename = path.name
        expected_rows = {'train.csv':132481, 'test.csv':87841, 'test_label.csv':87841,
            'swat2.csv':449919, 'swat_train2.csv':495000}[filename]
        sources.append(dict(id=original['id'], dataset=dataset, filename=filename,
            drive_file_id=original['drive_file_id'],
            url='https://drive.usercontent.google.com/download?export=download&confirm=t&id=' + original['drive_file_id'],
            preserved_previous_file=original['path'], preserved_previous_sha256=sha(path),
            expected_rows=expected_rows, allowed_column_counts=[2] if filename=='test_label.csv'
                else [25,26] if dataset=='PSM' else [52],
            citation='https://drive.google.com/drive/folders/1RaIJQ8esoWuhyphhmMaH-VCDh-WIluRR',
            publisher_checksum=None))
    if len(sources) != 5: raise ValueError('All three PSM and two SWaT failed payloads required')
    aria2 = Path('C:/Users/10175/AppData/Local/aria2/aria2-1.37.0-win-64bit-build1/aria2c.exe')
    if not aria2.is_file(): raise FileNotFoundError(aria2)
    files = [catalog_path, 'configs/datasets/mtsad_psm.json', 'configs/datasets/mtsad_swat.json',
        'experiments/runs/mtsad_protocol_audit/sources/shcl/original/README.md',
        'scripts/flow_matching/register_shcl_shared_data_retry_v1.py',
        'scripts/flow_matching/retry_shcl_shared_data_v1.py']
    config = dict(schema_version=1, id='fm_shcl_shared_data_retry_v1',
        paper_id='shcl', proxy='http://127.0.0.1:17890', aria2c=str(aria2),
        author_shared_folder='1RaIJQ8esoWuhyphhmMaH-VCDh-WIluRR', sources=sources,
        download_root='F:/aicoding/IIA_Data/public_datasets/flow_matching/shcl_shared_data_retry_v1',
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in files],
        boundary='This folder is linked by both SHCL and DCdetector. Original 21-byte errors are retained. '
            'Every retry must pass a full CSV structure audit and independent local SHA256 reread. '
            'No publisher checksum is advertised. Native SHCL loader formats/preprocessing, data version '
            'and full original experiments remain separate obligations.')
    exclusive(ROOT / 'configs/acquisition/fm_shcl_shared_data_retry.v1.json', config)
    print(json.dumps(dict(registered_missing_files=len(sources), raw_error_files_preserved=True)))


if __name__ == '__main__':
    main()
