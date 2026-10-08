"""Reconcile code-completion evidence into config-owned ARA/resource bindings.

No training, source downloading or result reclassification occurs here. A core
reconstruction remains a core, and historical environments remain unvalidated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.literature.build_flow_matching_ara import ROOT, read, write
from scripts.flow_matching.audit_code_coverage import CODE_EXTENSIONS

NATIVE={'grasp','dfm','prismflow','cfm_ts'}
COMPARATORS={'dt_la','kgl','shcl','maelnet','pi_transformer'}
SUPPLEMENTAL={'jfi','flow_mismatching','moment_long_context','psm','omnianomaly','sensitivehue'}
OBJECTIVE_PAPERS={'independent':'ot_cfm','ot':'ot_cfm','sb':'ot_cfm','vp':'ot_cfm',
                  'target':'flow_matching','rectified':'rectified_flow','sf2m':'sf2m'}
SOURCE_AREAS=('fm_code_completion','fm_baseline_code_completion',
              'fm_foundation_baseline_sources','fm_tsad_baseline_sources',
              'fm_imputation_baseline_sources')


def paper_id(config):
    if config.get('paper_id'):return config['paper_id']
    ident=config['id']
    if ident.startswith('fm_objective_'):return OBJECTIVE_PAPERS[config['parameters']['objective']]
    for prefix,papers in [('fm_native_',NATIVE),('fm_comparator_',COMPARATORS),('fm_supplemental_',SUPPLEMENTAL)]:
        if ident.startswith(prefix):
            for name in sorted(papers,key=len,reverse=True):
                if ident[len(prefix):]==name or ident[len(prefix):].startswith(name+'_'):return name
    raise ValueError('Unmapped code configuration: '+ident)


def register(root=ROOT):
    ara_path=root/'configs/reproducibility/flow_matching_ara.v1.json'
    coverage_path=root/'configs/reproducibility/fm_code_coverage.v1.json'
    ara=read(ara_path);coverage=read(coverage_path);papers={p['id']:p for p in ara['papers']}
    resources={r['id']:r for r in coverage['source_resources']}
    for area in SOURCE_AREAS:
        for path in sorted((root/'experiments/runs'/area/'sources').glob('*/snapshot.json')):
            snapshot=read(path);directory=root/snapshot['original_path']
            if not any(p.is_file() and p.suffix.lower() in CODE_EXTENSIONS for p in directory.rglob('*')):continue
            ident=path.parent.name+'_completion'
            resources[ident]={'id':ident,'path':directory.relative_to(root).as_posix(),
                             'repository':snapshot['repository'].removeprefix('https://github.com/'),
                             'commit':snapshot['commit'],'archive_sha256':snapshot.get('archive_sha256',snapshot.get('sha256')),
                             'provenance':path.relative_to(root).as_posix()}
    coverage['source_resources']=list(resources.values())
    for ident, entries in coverage['local_entrypoints'].items():
        coverage['local_entrypoints'][ident]=[entry for entry in entries
            if entry.startswith(('iia_benchmark.models.','scripts.'))]
    bindings={'maelnet':['maelnet'],'psm':['psm'],'omnianomaly':['omnianomaly','omnianomaly_zhusuan','omnianomaly_tfsnippet'],
              'sensitivehue':['sensitivehue'],'usad':['usad_author'],'anomaly_transformer':['anomaly_transformer_author'],
              'tab':['merlion','tods','pyod']}
    for paper,names in bindings.items():
        current=coverage['paper_source_bindings'].setdefault(paper,[])
        for name in names:
            ident=name+'_completion'
            if ident not in resources:raise ValueError('Missing expected snapshot: '+ident)
            if ident not in current:current.append(ident)
    native=read(root/'docs/reports/fm_native_code_completion_2026-10-08.json')
    supplemental=read(root/'docs/reports/fm_supplemental_code_completion_2026-10-08.json')
    details={p['id']:p for p in native['papers']}
    details.update({p['paper_id']:p for p in supplemental['papers']})
    for ident in NATIVE|{'jfi','flow_mismatching','moment_long_context'}:
        status='local_paper_component_reconstruction_with_gaps' if ident in {'flow_mismatching','moment_long_context'} else 'local_paper_core_reconstruction_with_gaps'
        item=details[ident]
        note='代码、主消融或组件已按原文还原；仍缺：'+'；'.join(item['missing'])
        coverage['main_status_overrides'][ident]={'status':status,'note':note}
        papers[ident]['reproduction_gap']=note
    descriptions={
      'maelnet':('source_present_not_end_to_end_verified','原文链接官方ModMaelNet源码、CPU核心前后向和奖励/去慢学习器变体在场；完整RL、checkpoint及协议验收待做。'),
      'dt_la':('local_paper_core_reconstruction_with_gaps','MRH、双编码器、注意力熵、scaled-softmax及fit/score和显式消融已还原；未公开架构/归约/验证搜索细节待核。'),
      'kgl':('local_paper_core_reconstruction_with_gaps','GAT、cubic B-spline KAN、LSTM与去模块消融已还原；论文未公开训练目标/输出头/图邻域，本地选择显式登记，非完整原版。'),
      'shcl':('author_source_and_local_reconstruction_with_gaps','作者VAE源码保留，五骨干Transformer/VAE/CNN/RNN/LSTM及THM/连续性/高斯KL/评分和部分消融已补；adaptive masking及精确架构仍待对齐。'),
      'psm':('source_present_legacy_environment_pending','官方RANSynCoders源码及同步开关在场；历史TensorFlow环境、私有生产模型与BKPI、全消融及TAB适配待验收。'),
      'omnianomaly':('source_present_legacy_environment_pending','官方OmniAnomaly及ZhuSuan/tfsnippet源码在场；TensorFlow1.12/TFP0.5历史环境与精确版本/消融对齐待验收。'),
      'sensitivehue':('source_present_not_end_to_end_verified','官方源码及CPU模型前后向已验证；Table4九行损失/结构变体已实现，论文公式与发布源码的均值缩放差异分开记录；完整训练及数值对齐待验收。'),
      'pi_transformer':('third_party_mirror_present','镜像源码的CPU构造、phase reshape、causal mask与prior支持四项修复及single-head配置已物化；作者现仓库仍仅README/LICENSE，完整期刊等价性待核。')}
    for ident,(status,note) in descriptions.items():
        coverage['main_status_overrides'][ident]={'status':status,'note':note};papers[ident]['reproduction_gap']=note
    for ident in ('maelnet','psm','omnianomaly','sensitivehue'):
        resource=resources[bindings[ident][0]+'_completion'];papers[ident]['code_url']='https://github.com/'+resource['repository']
    for ident in ('usad','anomaly_transformer'):
        note=coverage['main_status_overrides'][ident]['note']
        addition=' 官方原始源码现已补入，但本地实现与原版的完整等价性仍待核验。'
        if addition not in note:coverage['main_status_overrides'][ident]['note']=note+addition
    # Table2 alone did not establish the user's LSTM claim. Primary Table7 does.
    shcl=papers['shcl']
    shcl_rows={'SHCL-Transformer':[95.06,98.97,92.07,97.60,98.81],
               'SHCL-VAE':[93.75,99.19,91.78,84.22,96.48],
               'SHCL-CNN':[78.81,91.41,79.78,66.31,89.39],
               'SHCL-RNN':[85.75,93.16,70.64,73.34,89.47],
               'SHCL-LSTM':[71.84,96.42,83.51,73.54,88.49]}
    shcl['reported_results']=[r for r in shcl['reported_results'] if str(r.get('table'))!='7']
    for method,values in shcl_rows.items():
        for dataset,value in zip(('MSL','PSM','SMD','SMAP','SWaT'),values):
            shcl['reported_results'].append({'pdf_page':12,'table':'7','metric':'F1_percent','dataset':dataset,
                'value':value,'method_row':method,'status':'author_reported_not_locally_reproduced',
                'verification':'primary_pdf_text_table_review',
                'protocol':'paper validation/adjustment protocol; not established TAB raw-F1 equivalence'})
    shcl['table_caveat']='Table7 confirms all five backbones and SHCL-LSTM PSM96.42. Earlier Table2-only audit omitted Table7; 2026-10-08 correction. Original paper evaluation remains separate from TAB and strict deployment.'
    config_paths=[]
    for pattern in ('fm_native*.json','fm_comparator*.json','fm_supplemental*.json','fm_objective*.json'):
        config_paths+=sorted((root/'configs/models').glob(pattern))
    for path in config_paths:
        config=read(path)
        if config.get('entrypoint') and not config['entrypoint'].startswith('iia_benchmark.models.'):
            config['author_entrypoint']=config.pop('entrypoint')
            config['entrypoint_scope']='author_source_requires_isolated_runtime'
            write(path,config)
        if 'params' in config:
            if 'parameters' in config and config['parameters']!=config['params']:
                raise ValueError('Ambiguous parameter dictionaries: '+path.name)
            config['parameters']=config.pop('params')
            write(path,config)
        ident=paper_id(config);relative=path.relative_to(root).as_posix()
        mappings={m['path']:m for m in papers[ident]['model_configs']}
        mappings[relative]={'path':relative,'configuration':config}
        papers[ident]['model_configs']=list(mappings.values())
        if config.get('entrypoint'):
            entries=coverage['local_entrypoints'].setdefault(ident,[])
            if config['entrypoint'] not in entries:entries.append(config['entrypoint'])
            module,symbol=config['entrypoint'].split(':',1)
            source='src/'+Path(*module.split('.')).with_suffix('.py').as_posix()
            locator={'variant':config['id'],'path':source,'markers':[symbol]}
            locators=coverage['variant_code_locators'].setdefault(ident,[])
            locators[:]=[v for v in locators if v['variant']!=config['id']];locators.append(locator)
    coverage['completion_reports']=['docs/reports/fm_native_code_completion_2026-10-08.json',
                                  'docs/reports/fm_comparator_code_completion_2026-10-08.json',
                                  'docs/reports/fm_supplemental_code_completion_2026-10-08.json',
                                  'docs/reports/fm_baseline_code_completion_2026-10-08.json']
    coverage['all_methods_complete']=False;coverage['all_ablations_complete']=False
    write(ara_path,ara);write(coverage_path,coverage)
    project_path=root/'configs/projects/flow_matching_research.v1.json';project=read(project_path)
    additions=['configs/acquisition/fm_'+group+'_code_completion_2026-10-08.json' for group in ('native','comparator','supplemental','baseline')]
    for group in ('foundation','tsad','imputation'):
        registry='configs/acquisition/fm_'+group+'_baseline_sources_2026-10-08.json'
        report='docs/reports/fm_'+group+'_baseline_sources_2026-10-08.json'
        if (root/registry).exists():additions.append(registry)
        if (root/report).exists() and report not in coverage['completion_reports']:
            coverage['completion_reports'].append(report)
    write(coverage_path,coverage)
    project['acquisition_registries']=list(dict.fromkeys(project['acquisition_registries']+additions))
    project['evidence_reports']=list(dict.fromkeys(project['evidence_reports']+coverage['completion_reports']))
    write(project_path,project)
    return {'code_resources':len(resources),'new_configuration_profiles':len(config_paths),
            'all_paper_code_complete':False,'all_paper_ablations_complete':False}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(register(),ensure_ascii=True,indent=2))


if __name__=='__main__':main()
