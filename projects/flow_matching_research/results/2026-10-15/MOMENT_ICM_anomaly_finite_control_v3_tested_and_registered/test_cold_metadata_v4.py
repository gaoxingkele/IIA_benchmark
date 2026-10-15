import copy
import os
from pathlib import Path
import subprocess
import sys
import pytest
from scripts.flow_matching import run_moment_icm_finite_control_cold_metadata_v4 as p


def test_actual_pre_guard_leaves_resource_api_cold_and_installable_in_fresh_process():
    code=('import sys;from scripts.flow_matching.run_moment_icm_finite_control_cold_metadata_v4 import registration,pre_guard;'
        'r=registration();pre_guard(r);'
        'from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades;'
        'a=install_facades();a["resource_snapshot"]();'
        'assert not any(x in sys.modules for x in ["numpy","pandas","torch"])')
    result=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=p.ROOT,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr[-4000:]


@pytest.mark.parametrize('mutation',['worker','guard','lock'])
def test_cold_metadata_repair_cannot_replace_the_scientific_worker_or_relax_guards(mutation):
    runtime=p.registration();p.pre_guard(runtime)
    if mutation=='worker':runtime['worker_configuration']='other'
    elif mutation=='guard':runtime['settings']['minimum_commit_headroom_bytes']=1
    else:runtime['resource_lock']='other'
    with pytest.raises(ValueError):p.pre_guard(runtime)


def test_an_existing_scientific_launch_or_artifact_is_never_retried_as_metadata(tmp_path):
    base=tmp_path/'old';base.mkdir();artifact=tmp_path/'artifact'
    (base/'controller.stderr.log').write_text('Install resource facades only in a fresh cold metadata interpreter',encoding='utf-8')
    p.require_pre_science_failure(base,artifact)
    (base/'worker_launch.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError):p.require_pre_science_failure(base,artifact)
    (base/'worker_launch.json').unlink();artifact.mkdir()
    with pytest.raises(ValueError):p.require_pre_science_failure(base,artifact)
