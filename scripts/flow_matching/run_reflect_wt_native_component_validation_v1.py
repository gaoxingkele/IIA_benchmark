"""Scoped OpenCV dependency path for original component tests; no global mutation."""
import argparse
import os
import subprocess
from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT,read,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--environment',required=True)
    args=parser.parse_args();cfg=read(ROOT/args.environment)
    if sha(cfg['python'])!=cfg['python_sha256'] or not (ROOT/cfg['isolated_target']/'cv2').is_dir():
        raise ValueError('Exact native component interpreter and verified isolated cv2 required')
    report=read(ROOT/cfg['output_directory']/'isolated_opencv_dependency_installation.json')
    if not report['passed'] or report['configuration_sha256']!=sha(ROOT/args.environment):
        raise ValueError('Verified exact isolated native prerequisite required')
    environment={**os.environ,'PYTHONUTF8':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'PYTHONPATH':os.pathsep.join([str((ROOT/cfg['isolated_target']).resolve()),str(ROOT/'src'),str(ROOT)])}
    command=[cfg['python'],'-X','utf8','-m','pytest','-q',
        'tests/test_fm_reflect_wt_native_components_v1.py',
        'tests/test_fm_flow_mismatching_optimizer_source_acquisition_v1.py',
        'tests/test_fm_flow_mismatching_native_guidance_sources_v1.py']
    return subprocess.run(command,cwd=ROOT,env=environment).returncode


if __name__=='__main__':raise SystemExit(main())
