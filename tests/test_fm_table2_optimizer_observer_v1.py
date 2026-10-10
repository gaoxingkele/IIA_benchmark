"""A unit fixture checks optimizer observation; it is never a scientific result."""
import json
import os
from pathlib import Path
import subprocess

import torch

from scripts.flow_matching.spectral_table2_bootstrap_v1 import sha
from scripts.flow_matching.run_light_controller import ROOT


def test_generator_counter_excludes_real_nested_evaluator_optimizer(tmp_path):
    metrics=tmp_path/'metrics';metrics.mkdir();(metrics/'__init__.py').write_text('')
    for module,function in [('discriminative_metrics','discriminative_score_metrics'),
        ('predictive_metrics','predictive_score_metrics2'),('context_fid','context_fid')]:
        (metrics/(module+'.py')).write_text('def '+function+'(real,fake): return 0.0\n')
    (tmp_path/'nested.py').write_text('import torch\n'
        'def evaluate():\n'
        ' net=torch.nn.Linear(2,1); opt=torch.optim.Adam(net.parameters())\n'
        ' for _ in range(3):\n'
        '  opt.zero_grad(); net(torch.ones(2,2)).sum().backward(); opt.step()\n')
    entry=tmp_path/'author_fixture_only.py'
    entry.write_text('import torch\nfrom types import SimpleNamespace\nfrom nested import evaluate\n'
        'args=SimpleNamespace(seed=42); net=torch.nn.Linear(2,1); optimizer=torch.optim.AdamW(net.parameters())\n'
        'for nb_iter in range(1,5):\n'
        ' optimizer.zero_grad(); net(torch.ones(2,2)).sum().backward(); optimizer.step(); evaluate()\n')
    contract=dict(entry=str(entry),entry_sha256=sha(entry),workspace=str(tmp_path),arguments=[],
        stage='training',optimizer_updates=4,model_keys=['net'],receipt=str(tmp_path/'receipt.json'),
        final_checkpoint=str(tmp_path/'final.pth'))
    path=tmp_path/'contract.json';path.write_text(json.dumps(contract))
    child=subprocess.run([str(ROOT/'.venv/spectral-table2-author-py310/Scripts/python.exe'),'-X','utf8','-m',
        'scripts.flow_matching.spectral_table2_bootstrap_v1','--contract',str(path)],cwd=ROOT,
        env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True,text=True,encoding='utf-8',timeout=45)
    assert child.returncode==0,child.stderr
    receipt=json.loads((tmp_path/'receipt.json').read_text())
    assert receipt['optimizer_updates']==4
    saved=torch.load(tmp_path/'final.pth',map_location='cpu',weights_only=False)
    assert saved['optimizer_updates']==4 and saved['final_iteration']==4
    assert {float(s['step']) for s in saved['optimizer']['state'].values()}=={4.}
