"""Publication layout only; preserve the frozen v1 analysis and source bytes."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.flow_matching.summarize_verified_protocol_effects_v1 import ROOT,read,sha,write,CONFIG


def main():
    config=read(ROOT/CONFIG);folder=ROOT/config['output_directory'];source=folder/'report.json';report=read(source)
    if (folder/'protocol_effects_v2.png').exists():raise FileExistsError('Preserve prior publication rendering')
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'verified_protocol_effects_publication_v2'})
    fig,(ax,curve)=plt.subplots(1,2,figsize=(13.6,6.3),gridspec_kw={'width_ratios':[1.15,1]})
    y=np.arange(4);cases=report['cases'];h=.32
    ax.barh(y-h/2,[100*r['same_threshold_effect']['no_PA_f1'] for r in cases],height=h,label='No point adjustment',color='#176482')
    ax.barh(y+h/2,[100*r['same_threshold_effect']['PA_f1'] for r in cases],height=h,label='PA (uses test event labels)',color='#e2983a')
    ax.set(yticks=y,yticklabels=[r['label'] for r in cases],xlim=(0,110),xlabel='F1 (%)',title='Same model, scores and threshold within each row')
    ax.set_ylim(4.05,-.65);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    for i,r in enumerate(cases):
        for offset,key in [(-h/2,'no_PA_f1'),(h/2,'PA_f1')]:
            value=100*r['same_threshold_effect'][key];ax.text(value+.7,i+offset,f'{value:.2f}',va='center',fontsize=9)
    ax.legend(loc='lower right',fontsize=8)
    sb=cases[-1];grid=sb['diagnostic_test_pooled_grid'];x=[r['ratio_percent'] for r in grid]
    curve.plot(x,[100*r['f_score'] for r in grid],'o-',color='#176482',label='No PA: test-pooled ratio grid')
    curve.plot(x,[100*r['adjust_f_score'] for r in grid],'o-',color='#e2983a',label='PA: same ratio grid')
    curve.axhline(100*sb['primary_validation_only_no_PA_f1'],ls='--',color='#555555',label='Frozen validation-only primary (no PA)')
    curve.set(xscale='log',ylim=(0,100),xlabel='Predeclared test-pooled ratio (%)',ylabel='F1 (%)',title='One unchanged local SB model / SWaT')
    curve.grid(alpha=.18);curve.legend(loc='upper left',bbox_to_anchor=(0,-.21),fontsize=8)
    fig.suptitle('Evaluation-protocol effects in four verified cases — not an algorithm ranking',fontsize=13)
    fig.text(.5,.025,'Pi: native + independent replay. MOMENT: saved-score audit. SB: native replay. Grid/test-label selection is diagnostic.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.185,right=.985,top=.82,bottom=.30,wspace=.32)
    paths=[]
    for extension in ['png','svg']:
        path=folder/('protocol_effects_v2.'+extension);fig.savefig(path,dpi=170,metadata={'Creator':'IIA benchmark protocol evidence report'});paths.append(path)
    plt.close(fig)
    write(folder/'publication_render_v2_receipt.json',dict(passed=True,source_report_sha256=sha(source),
        source_renderer_sha256=sha(Path(__file__)),output_sha256={p.name:sha(p) for p in paths},
        numerical_values_or_cases_changed=False,frozen_v1_source_and_outputs_changed=False,
        correction='Place the ratio-grid legend outside the data axes; keep all points and both curve peaks visible.',
        preferred_publication_files=[p.name for p in paths],new_training_or_inference=0,all_original_experiments_complete=False))
    print(paths[0])


if __name__=='__main__':main()
