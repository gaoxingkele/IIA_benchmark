"""Original zero-shot worker; only short-source window assembly is adapted."""
from __future__ import annotations
from types import FunctionType
import numpy as np
from scripts.flow_matching import run_tsad_experiment as original
from iia_benchmark.models.fm_moment_masked_context_v1 import MaskedContextWindows


def covering_windows(values,window):
    if window!=512:raise ValueError('Masked worker is exclusively for original512 MOMENT')
    if values.ndim==2 and 8<=len(values)<window:
        return MaskedContextWindows(values,window),np.array([0],dtype=np.int64)
    return original.covering_windows(values,window)


def align_scores(scores,starts,length,window):
    if window!=512:raise ValueError('Original512 context required')
    if length<window:
        scores=np.asarray(scores)
        if scores.shape!=(1,length) or not np.array_equal(starts,[0]) or not np.isfinite(scores).all():
            raise ValueError('Every real source point scored once; no fill targets allowed')
        return scores[0].astype(float)
    return original.align_scores(scores,starts,length,window)


def run(root,job,settings):
    config=original.load_model_config(root/job['model_config'])
    if (config['entrypoint']!='iia_benchmark.models.fm_moment_masked_context_v1:MomentMaskedContextWindow'
            or config['parameters']['epochs']!=0 or job['window']!=512):
        raise ValueError('Only frozen official zero-shot512 MOMENT may use this worker')
    def write_result(path,result):
        result['native_short_context_protocol']=dict(context_points=512,
            fill_observed=False,fill_scored=False,fit_points_used_for_parameter_estimation=0,
            complete_real_timestamps_scored=True,long_context_numerical_worker_unchanged=True)
        result['boundary']+=' Explicit native observation masks for complete short sources;all fill points excluded from scores. New protocol,not historical-paper equivalence.'
        return original.write_json(path,result)
    worker=FunctionType(original.run.__code__,dict(original.run.__globals__,
        covering_windows=covering_windows,align_scores=align_scores,write_json=write_result))
    return worker(root,job,settings)


def main():
    return FunctionType(original.main.__code__,dict(original.main.__globals__,run=run))()


if __name__=='__main__':main()
