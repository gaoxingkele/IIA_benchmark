"""Execute the original and repaired native plot on the actual failed trace."""
import argparse
import os
from pathlib import Path
import shutil
import sys

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha
from scripts.flow_matching.audit_adopted_gan_native_v1 import verify_source
from scripts.flow_matching.rcgan_plot_trace_compat_v1 import install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    queue = read(ROOT / args.queue)
    from scripts.flow_matching.run_rcgan_native_original_queue_v2 import verify_registration
    verify_registration(queue)
    if Path(sys.executable).resolve() != Path(queue['python']).resolve():
        raise ValueError('Designated native interpreter required')
    registry = read(ROOT / queue['source_registry'])
    asset = next(a for a in registry['assets'] if a['id'] == 'rcgan')
    before_source = verify_source(asset)
    previous = read(ROOT / queue['prior_attempt_queue'])
    original = ROOT / previous['jobs'][0]['output_directory'] / 'experiments/traces/rcgan_mnistfull__seed1103.trace.txt'
    before_trace = sha(original)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    traces = output / 'experiments/traces'
    traces.mkdir(parents=True)
    copy = traces / original.name
    shutil.copyfile(str(original), str(copy))
    os.chdir(str(output))
    sys.path.insert(0, asset['original_directory'])
    import plotting
    import numpy as np
    from pandas import read_table
    frame = read_table(str(copy), sep=' ')
    if len(frame) != 1 or int(frame['epoch'].iloc[0]) != 0 or not {'pdf', 'real_pdf'}.issubset(frame.columns):
        raise ValueError('Actual first full author training trace required')
    numpy_before = np.random.get_state()
    original_failed = False
    try:
        plotting.plot_trace('rcgan_mnistfull__seed1103', xmax=500, dp=False)
    except KeyError as error:
        if 'll' not in str(error) or 'real_ll' not in str(error):
            raise
        original_failed = True
        failure = str(error)
    finally:
        plotting.plt.close('all')
    if not original_failed:
        raise ValueError('Original actual compatibility failure must be reproduced')
    receipt = install(plotting)
    patched = frame.reindex(columns=['epoch', 'll', 'real_ll'])
    if not np.array_equal(patched['epoch'].values, frame['epoch'].values) or not patched[['ll', 'real_ll']].isna().all().all():
        raise ValueError('Missing likelihoods must remain missing')
    if plotting.plot_trace('rcgan_mnistfull__seed1103', xmax=500, dp=False) is not True:
        raise ValueError('Full original trace plotting function must complete')
    numpy_after = np.random.get_state()
    if not all(np.array_equal(a, b) for a, b in zip(numpy_before, numpy_after)):
        raise ValueError('Plot compatibility must not consume training randomness')
    artifacts = [traces / 'rcgan_mnistfull__seed1103_trace.png', traces / 'rcgan_mnistfull__seed1103_trace.pdf']
    if not all(p.exists() and p.stat().st_size > 1000 for p in artifacts):
        raise ValueError('Both original plot output formats required')
    if sha(original) != before_trace or sha(copy) != before_trace or verify_source(asset) != before_source:
        raise ValueError('Original trace or source changed')
    write(ROOT / args.report, dict(passed=True, actual_full_training_trace_used=True,
        original_native_plot_failure_reproduced=True, failure=failure, original_trace_sha256=before_trace,
        original_trace_path=str(original), native_python=sys.executable, original_source_receipt=before_source,
        plotting_compatibility=receipt, missing_likelihoods_remain_nan=True,
        numpy_random_state_unchanged=True, original_trace_and_all_source_files_unchanged=True,
        artifacts=[dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in artifacts],
        benchmark_performance=False, full_training_completed=False, all_original_experiments_complete=False))


if __name__ == '__main__':
    main()
