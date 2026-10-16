"""Add the paper-text activation/weight-decay recipe to the unchanged full grid.

The author-logged Tanh recipe remains separately live. Paper LeakyReLU slope is
unspecified; .01 is an explicit local assumption, not recovered author metadata.
"""
import copy
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha

MODULE='scripts.flow_matching.run_trajectorynet_EB_paper_text_full180_v1'
QUEUE='configs/experiments/fm_TrajectoryNet_EB_paper_text_full180.v1.json'
MODEL='configs/models/fm_TrajectoryNet_EB_paper_text_full_grid.v1.json'
BASE='scripts/flow_matching/run_trajectorynet_EB_native_full180_v1.py'
ARTIFACT='F:/aicoding/IIA_Data/experiments/TrajectoryNet_EB_paper_text_full180_v1'
STATE='experiments/runs/fm_TrajectoryNet_EB_paper_text_full180_v1'
SOURCE_ACQUISITION='configs/acquisition/fm_TrajectoryNet_arxiv_source.v1.json'


def isolated_driver():
    spec=importlib.util.spec_from_file_location('_TrajectoryNet_full_grid_paper_text_private_driver_v1',ROOT/BASE)
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    original_registration,original_runtime=driver.registration,driver.runtime
    driver.MODULE,driver.QUEUE,driver.MODEL=MODULE,QUEUE,MODEL
    def registration():
        c=original_registration()
        profile=read(ROOT/MODEL)
        acquisition=read(ROOT/SOURCE_ACQUISITION)
        material=Path(acquisition['output_directory'])/'verified_acquisition.json'
        receipt=read(material)
        if not receipt['passed'] or receipt['source_execution_performed']:
            raise ValueError('Complete official source archive must be verified without executing TeX')
        c['source_parameters']=dict(c['source_parameters'],nonlinearity=profile['native_activation_key'],
            weight_decay=profile['paper_text_weight_decay'])
        jobs=[]
        for original in c['jobs']:
            job=copy.deepcopy(original)
            job['id']=job['id'].replace('TrajectoryNet_EB100__','TrajectoryNet_EB100_papertext_Leaky001_WD5e5__',1)
            job['artifact_directory']=ARTIFACT+'/'+hashlib.sha256(job['id'].encode()).hexdigest()[:20]
            jobs.append(job)
        c.update(jobs=jobs,state_root=STATE,artifact_root=ARTIFACT,
            protocol_variant='paper_text_LeakyReLU_slope001_local_assumption_weight_decay5e5',
            activation_extension=dict(key=profile['native_activation_key'],negative_slope=profile['local_negative_slope'],
                author_negative_slope_known=False,original_source_activation_table_has_LeakyReLU=False),
            paper_text_citation=dict(arxiv_id='2002.04461v2',pdf_page=7,
                source_file='sections/6_experiments.tex',source_line=2),
            author_logged_profile_remains_separate='configs/experiments/fm_TrajectoryNet_EB_native_full180.v1.json')
        extra=[BASE,SOURCE_ACQUISITION,str(material),
            'tests/test_fm_trajectorynet_paper_text_full180_v1.py',
            'projects/flow_matching_research/ara/adopted_baselines/trajectorynet_paper_text_vs_logged_recipe.2026-10-16_v1.json']
        extra.extend(str(Path(receipt['text_directory'])/n) for n in [
            'icml.tex','sections/main.tex','sections/6_experiments.tex','sections/appendices.tex',
            'tables/eb_table.tex','tables/eb_table_v5.tex'])
        paths={r['path'] for r in c['source_receipts']}|set(extra)
        c['source_receipts']=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(paths)]
        c['boundary'] += ' This additional recipe follows the paper-text LeakyReLU and5e-5weightdecay,while retaining author64x3widths and an explicit local negative_slope=.01 because that slope is not specified. This is not certified exact historical paper execution. The original loggedTanh/1e-5recipe remains separate; their joint activation/decay difference cannot isolate either cause.'
        return c
    def runtime(c,job,output,checkpoint=None):
        # Only numerical children call this, after the same original admission
        # policy. The metadata controller never imports Torch.
        sys.path.insert(0,c['overlay_directory'])
        sys.path.insert(0,str(ROOT/c['original_source_root']))
        import torch
        activation=importlib.import_module('TrajectoryNet.lib.layers.odefunc')
        extension=c['activation_extension']
        # A process-local dictionary copy adds only the missing documented
        # activation. Author files and all existing activation objects stay intact.
        activation.NONLINEARITIES=dict(activation.NONLINEARITIES,
            **{extension['key']:torch.nn.LeakyReLU(negative_slope=extension['negative_slope'],inplace=False)})
        result=original_runtime(c,job,output,checkpoint)
        modules=[m for m in result[4].modules() if isinstance(m,torch.nn.LeakyReLU)]
        # Original builder reuses the same stateless activation object; modules()
        # deduplicates it. Inspect activation_fns slots to verify all3hidden layers.
        network=result[4].chain[0].odefunc.odefunc.diffeq
        slots=list(network.activation_fns)
        if (len(slots)!=3 or not all(isinstance(m,torch.nn.LeakyReLU) and m.negative_slope==.01 for m in slots)
                or not modules or result[3].weight_decay!=5e-5):
            raise ValueError('All three original hidden activations and paper5e-5weightdecay required')
        return result
    driver.registration,driver.runtime=registration,runtime
    return driver


driver=isolated_driver()
train=driver.train
replay=driver.replay


if __name__=='__main__':driver.main()
