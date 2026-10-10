"""Native source budgets and genuine Windows spawn boundary checks."""
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.flow_matching.register_spectral_table2_original_v1 import ROOT,source_pipelines
from scripts.flow_matching.run_spectral_table2_original_v1 import arguments_for,periodic_counts


def test_full_released_budgets_and_periodic_selection_are_retained():
    source=ROOT/'experiments/runs/fm_project_acquisition/sources/spectral_mean_flow/original/src_medium'
    pipelines=source_pipelines(source)
    assert len(pipelines)==6
    for (method,dataset),stages in pipelines.items():
        if method=='spectral_flow':
            assert stages[0]['optimizer_updates']==100000
            assert periodic_counts(stages[0])==Counter({i:5 for i in range(0,100001,2500)})
            assert '--sampling-timesteps' in stages[1]['original_arguments']
            assert stages[1]['original_arguments'][stages[1]['original_arguments'].index('--sampling-timesteps')+1]=='500'
        else:
            assert stages[0]['optimizer_updates']==(200999 if dataset=='mujoco' else 100999)
            assert stages[0]['warmup_updates']==999
            assert periodic_counts(stages[0])[999]==3
            assert stages[1]['optimizer_updates']==(62500 if dataset=='mujoco' else 20000)
        for stage in stages:
            actual=arguments_for(stage,1103)
            differences=[(x,y) for x,y in zip(stage['original_arguments'],actual) if x!=y]
            assert all(y in ('0','1103') for _,y in differences)
            if stage['stage']=='evaluation':assert actual==stage['original_arguments']


def test_top_level_author_body_executes_once_under_real_spawn(tmp_path):
    entry=tmp_path/'unguarded_author.py';marker=tmp_path/'executions.jsonl';result=tmp_path/'result.json'
    entry.write_text('import multiprocessing,os,json\n'
        f'with open({str(marker)!r},"a") as output: output.write(json.dumps(os.getpid())+"\\n")\n'
        'with multiprocessing.get_context("spawn").Pool(2) as pool: values=pool.map(abs,[-1,-2,-3])\n'
        f'with open({str(result)!r},"w") as output: json.dump(values,output)\n',encoding='utf-8')
    outer=tmp_path/'guarded_outer.py'
    outer.write_text('import sys\nfrom scripts.flow_matching.spectral_table2_bootstrap_v1 import execute_original_entry\n'
        'if __name__=="__main__":\n'
        ' original=sys.modules["__main__"]\n'
        f' execute_original_entry({str(entry)!r},[])\n'
        ' assert sys.modules["__main__"] is original\n',encoding='utf-8')
    completed=subprocess.run([sys.executable,'-X','utf8',str(outer)],cwd=ROOT,timeout=45,
        env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1'),
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert completed.returncode==0,completed.stderr
    assert len(marker.read_text().splitlines())==1
    assert json.loads(result.read_text())==[1,2,3]
