"""Freeze official Windows wheels for the released SHCL numerical dependencies."""
import html
import json
from pathlib import Path
import re
from urllib.parse import unquote,urlsplit
from urllib.request import ProxyHandler,build_opener

from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    proxy='http://127.0.0.1:17890'; opener=build_opener(ProxyHandler({'http':proxy,'https':proxy}))
    index='https://download.pytorch.org/whl/cu113/torch/'
    with opener.open(index,timeout=30) as response: content=response.read().decode('utf-8')
    filename='torch-1.11.0+cu113-cp310-cp310-win_amd64.whl'
    urls=[html.unescape(value) for value in re.findall(r'href="([^"]+)"',content)
        if unquote(urlsplit(html.unescape(value)).path).split('/')[-1]==filename]
    if len(urls)!=1: raise ValueError('Exactly one official Torch1.11 Windows/CUDA11.3 wheel required')
    url=urls[0]; digest=urlsplit(url).fragment.removeprefix('sha256=')
    if not re.fullmatch('[0-9a-f]{64}',digest): raise ValueError('Publisher Torch wheel SHA256 required')
    wheels=[dict(package='torch',version='1.11.0+cu113',filename=filename,
        url=url.split('#')[0],sha256=digest,source_index=index)]
    versions={'numpy':'1.26.4','pandas':'2.2.3','scikit-learn':'1.3.0','scipy':'1.15.1','tqdm':'4.65.0',
        'typing-extensions':'4.5.0','python-dateutil':'2.9.0.post0','pytz':'2025.2','tzdata':'2025.2',
        'six':'1.17.0','joblib':'1.3.2','threadpoolctl':'3.1.0','colorama':'0.4.6'}
    for package,version in versions.items():
        endpoint='https://pypi.org/pypi/'+package+'/'+version+'/json'
        with opener.open(endpoint,timeout=30) as response: metadata=json.load(response)
        candidates=[row for row in metadata['urls'] if row['packagetype']=='bdist_wheel'
            and (row['filename'].endswith('cp310-cp310-win_amd64.whl')
                or row['filename'].endswith('none-any.whl')) and not row['yanked']]
        if len(candidates)!=1: raise ValueError('One complete native/universal wheel required: '+package)
        row=candidates[0]
        wheels.append(dict(package=package,version=version,filename=row['filename'],url=row['url'],
            sha256=row['digests']['sha256'],expected_size=row['size'],source_index=endpoint))
    source='experiments/runs/mtsad_protocol_audit/sources/shcl/original/requirements.txt'
    base=ROOT/'tmp/python/cpython-3.10-windows-x86_64-none/python.exe'
    aria2=Path('C:/Users/10175/AppData/Local/aria2/aria2-1.37.0-win-64bit-build1/aria2c.exe')
    if not base.is_file() or not aria2.is_file(): raise FileNotFoundError('Existing Python3.10/aria2 required')
    files=[source,'configs/models/fm_shcl_released_vae_native.v1.json',
        'scripts/flow_matching/register_shcl_author_environment_v1.py',
        'scripts/flow_matching/prepare_shcl_author_environment_v1.py']
    config=dict(schema_version=1,id='shcl_author_py310_environment_v1',proxy=proxy,aria2c=str(aria2),
        base_python=str(base),base_python_sha256=sha(base),
        root='F:/aicoding/IIA_Data/environments/shcl_author_py310_v1',
        python='F:/aicoding/IIA_Data/environments/shcl_author_py310_v1/venv/Scripts/python.exe',wheels=wheels,
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in files],
        declared_author_versions={'torch':'1.11.0','numpy':'2.2.2','pandas':'2.2.3',
            'scikit-learn':'1.3.0','scipy':'1.15.1','tqdm':'4.65.0'},
        numerical_dependency_correction='Pin NumPy1.26.4 rather than declared2.2.2: original Torch1.11 '
            'and sklearn1.3 require the legacy NumPy ABI/interfaces. Every other imported numerical '
            'dependency retains the declared version. Unused requirements are not imported by the native pipeline.',
        boundary='Exact declared Torch1.11 Windows/CUDA11.3 wheel and native imported packages except '
            'explicit NumPy compatibility repair. Original paper-era environment remains unknown; '
            'environment construction does not prove reported backbone, data or score equivalence.')
    exclusive(ROOT/'configs/acquisition/fm_shcl_author_environment.v1.json',config)
    print(json.dumps(dict(wheels=len(wheels),torch=wheels[0]['version'],numpy=versions['numpy'])))


if __name__=='__main__':main()
