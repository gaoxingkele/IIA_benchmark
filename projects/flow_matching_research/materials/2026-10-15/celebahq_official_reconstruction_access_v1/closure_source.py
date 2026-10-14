"""Archive observed official access gates without counting HTML as data."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
BASE = 'projects/flow_matching_research/materials/2026-10-15/celebahq_official_reconstruction_access_v1'
CONFIG = 'configs/acquisition/fm_celebahq_official_reconstruction_inputs.2026-10-15_v1.json'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

probe_path = ROOT / 'tmp/celebahq_all31_official_access_probe_20261015.json'
source_path = ROOT / 'tmp/celebahq_official_source_and_local_input_audit_20261015.json'
html_path = ROOT / 'tmp/celebahq_official_folder_20261015.html'
code_path = ROOT / 'tmp/celebahq_official_dataset_tool_20261015.py'
probe, source = read(probe_path), read(source_path)
rows = probe['records']
expected = {f'deltas{i:05d}.zip' for i in range(0, 30000, 1000)} | {'image_list.txt'}
assert {r['name'] for r in rows} == expected and len(rows) == 31
assert len({r['drive_file_id'] for r in rows}) == 31
assert sum(bool(r.get('redirected_to_signin')) for r in rows) == 30
assert len([r for r in rows if r.get('error_type')]) == 1
assert not any(r.get('prefix_is_zip') or r['payload_fully_downloaded'] for r in rows)
assert probe['actual_complete_downloads'] == probe['reconstructed_original_images'] == 0
assert sha(html_path) == probe['official_folder_html_sha256']
assert sha(code_path) == source['source_sha256']
tree = ast.parse(code_path.read_text(encoding='utf-8'))
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'create_celebahq')
assignments = {n.targets[0].id: n.value.value for n in ast.walk(function)
               if isinstance(n, ast.Assign) and len(n.targets) == 1
               and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}
assert assignments['expected_images'] == source['required_images'] == 202599
assert assignments['expected_zips'] == source['required_delta_zips'] == 30
assert source['required_pillow'] == '3.1.1' and source['required_libjpeg'] == '8d'
assert source['local_aligned_image_count'] == 202599
assert source['local_aligned_archive_md5'] == '00d2c5bc6d35e252742224ab0c1e8fcb'
assert source['local_aligned_first_image_dimensions'] == [178, 218]
assert not source['aligned_images_satisfy_official_unaligned_input']
asset = (ROOT / 'data/public_datasets/flow_matching/celeba_hq').resolve()
assert asset.is_relative_to(Path('F:/aicoding/IIA_Data').resolve())
metadata = asset / 'official_reconstruction_20261015_v1/metadata'
folder = ROOT / BASE
assert not folder.exists() and not metadata.exists() and not (ROOT / CONFIG).exists()
metadata.mkdir(parents=True)
shutil.copyfile(html_path, metadata / 'official_delta_folder.html')
shutil.copyfile(code_path, metadata / 'official_dataset_tool.py')
folder.mkdir(parents=True)
shutil.copyfile(probe_path, folder / 'official_all31_access_verification.json')
shutil.copyfile(source_path, folder / 'official_recipe_and_local_aligned_input_verification.json')
(folder / 'closure_source.py').write_bytes(Path(__file__).read_bytes())
(folder / '.gitattributes').write_text('*.py -text\n*.json -text\n*.md -text\n', encoding='utf-8')
sources = []
for row in rows:
    sources.append(dict(id='celebahq_original_' + row['name'].replace('.', '_'),
        filename=row['name'], drive_file_id=row['drive_file_id'],
        url='https://drive.google.com/file/d/' + row['drive_file_id'] + '/view',
        source_folder=probe['official_folder'],
        path=str(asset / 'official_reconstruction_20261015_v1/deltas' / row['name']),
        format='zip' if row['name'].endswith('.zip') else 'txt', checksum=None,
        manual_download=True, download_verified=False,
        observed_access='login_redirect' if row.get('redirected_to_signin') else 'transport_error_pending_recheck'))
pins = [metadata / 'official_delta_folder.html', metadata / 'official_dataset_tool.py',
        folder / 'closure_source.py', folder / 'official_all31_access_verification.json',
        folder / 'official_recipe_and_local_aligned_input_verification.json']
config = dict(schema_version=1, id='fm_celebahq_original_reconstruction_inputs_v1',
    checked_at_utc=datetime.now(timezone.utc).isoformat(),
    official_repository_commit=source['official_repository_commit'], official_recipe_url=source['source_url'],
    official_delta_folder=probe['official_folder'], asset_root=str(asset),
    proxy='http://127.0.0.1:17890', download_backend_preference='aria2c',
    automatic_download_enabled=False, sources=sources,
    required_original_inputs=dict(unaligned_image_count=202599, unaligned_image_layout='img_celeba/*.jpg',
        landmarks_layout='Anno/list_landmarks_celeba.txt', delta_zip_count=30,
        image_list='image_list.txt', pillow='3.1.1', libjpeg='8d',
        decoded_first_unaligned_image_md5=source['first_unaligned_jpeg_decoded_md5']),
    source_receipts=[dict(path=str(p), sha256=sha(p)) for p in pins],
    existing_aligned_archive_preserved=True, existing_aligned_images_are_not_original_unaligned_inputs=True,
    actual_new_complete_dataset_downloads=0, reconstructed_original_images=0,
    original_reconstruction_completed=False, all_materials_complete=False,
    original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
    all_original_experiments_complete=False,
    boundary='Public listing is accessible; 30 payload probes redirect to sign-in and one has a transport error. '
             'The error remains pending recheck. No HTML or partial response is counted as dataset data. '
             'This is an input/access audit, not a completed dataset reconstruction or experiment result.')
(ROOT / CONFIG).write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
for pin in config['source_receipts']:
    assert sha(pin['path']) == pin['sha256']
report = dict(passed=True, configuration=CONFIG, configuration_sha256=sha(ROOT / CONFIG),
    official_required_file_identities_verified=31, login_redirects=30, pending_transport_errors=1,
    pinned_official_recipe_requirements_checked=True, existing_aligned_input_checksum_verified=True,
    metadata_source_hashes_verified=True, actual_new_complete_dataset_downloads=0,
    reconstructed_original_images=0, all_materials_complete=False, all_original_experiments_complete=False)
(folder / 'closure_validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
lines = ['Official CelebA-HQ reconstruction input/access audit', '',
    'The pinned official recipe requires 202599 unaligned original images, landmarks, 30 delta ZIPs, '
    'image_list.txt, Pillow3.1.1 and libjpeg8d. Existing aligned CelebA ZIP MD5 and 202599 image entries '
    'were verified; its first image is178x218 and it does not satisfy the unaligned-input requirement.', '',
    'The official shared folder lists every required delta filename. Thirty payload probes redirect to '
    'Google sign-in. One transport error remains pending recheck. No completed dataset payload or '
    'reconstructed image was produced. Existing raw files, partial transfers and scientific workers were preserved.', '',
    'The complete official file identities and expected local input paths are in the configuration: ' + CONFIG, '',
    'Official pinned recipe: ' + source['source_url'], 'Official delta folder: ' + probe['official_folder']]
(folder / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps(report))
