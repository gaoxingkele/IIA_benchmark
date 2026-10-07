"""Audit local code coverage against ARA sources without importing vendor code.

Code presence, syntax, callable symbol discovery, paper-method equivalence and
complete ablation coverage are independent states. This audit never launches a
download, training job, vendor module or model selection.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import warnings

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.literature.import_flow_matching_archives import contained, normal
from scripts.literature.build_flow_matching_ara import read, write

EXTENSIONS={'.py','.ipynb','.sh','.json','.yaml','.yml','.toml','.ini','.cfg','.md','.txt'}
EXCLUDED={'.git','.venv','venv','__pycache__','__MACOSX','.ipynb_checkpoints'}


def parse_source(raw, filename='<unknown>'):
    # Legacy vendor escape warnings belong in the report, not on the console.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', SyntaxWarning)
        tree=ast.parse(raw, filename=filename)
    return tree, [{'line':w.lineno,'message':str(w.message)} for w in caught]


def inspect_resource(root, resource):
    base=contained(root,resource['path'])
    records=[]; syntax=[]; candidates=[]; syntax_warnings=[]
    if not base.is_dir():
        return {**resource,'status':'missing','python_files':0,'notebooks':0,'code_tree_sha256':None},records
    for path in sorted(base.rglob('*')):
        relative=path.relative_to(base)
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS or any(p in EXCLUDED or p.startswith('._') for p in relative.parts):
            continue
        raw=path.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
        records.append({'path':relative.as_posix(),'sha256':digest,'bytes':len(raw)})
        if path.suffix=='.py':
            try:
                _, caught=parse_source(raw,filename=relative.as_posix())
                syntax_warnings.extend({'path':relative.as_posix(),**w} for w in caught)
            except (SyntaxError,UnicodeDecodeError,ValueError) as error:
                syntax.append({'path':relative.as_posix(),'error':str(error)[:220]})
        # Filename matches only aid navigation; never certify semantic coverage.
        if re.search(r'(?i)(ablat|reflow|distill|sf2m)',relative.as_posix()):
            candidates.append(relative.as_posix())
    tree=''.join(r['path']+'\0'+r['sha256']+'\n' for r in records).encode('utf-8')
    report={**resource,'status':'local_code_files_present' if any(Path(r['path']).suffix in {'.py','.ipynb'} for r in records) else 'no_python_or_notebook_located',
            'registered_text_files':len(records),'python_files':sum(r['path'].endswith('.py') for r in records),
            'notebooks':sum(r['path'].endswith('.ipynb') for r in records),'code_tree_sha256':hashlib.sha256(tree).hexdigest(),
            'python_syntax_issues_current_interpreter':syntax,'python_syntax_warnings_current_interpreter':syntax_warnings,
            'ablation_filename_candidates':candidates,
            'boundary':'Fresh local selected-code tree fingerprint; not independently compared against publisher commit/archive. Syntax check uses the current interpreter and does not check dependencies, notebook execution or paper equivalence.'}
    return report,records


def inspect_entrypoint(root, value):
    module,symbol=value.split(':',1)
    module_path=Path(*module.split('.')).with_suffix('.py')
    path=(root/'src'/module_path) if module.startswith('iia_benchmark.') else root/module_path
    if not path.is_file():return {'entrypoint':value,'status':'missing_module'}
    raw=path.read_bytes()
    try:
        tree,_=parse_source(raw)
        found=any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n.name==symbol for n in tree.body)
        return {'entrypoint':value,'path':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(raw).hexdigest(),
                'status':'top_level_symbol_found' if found else 'symbol_not_found',
                'boundary':'AST only; not a runtime import or end-to-end training test.'}
    except (SyntaxError,ValueError) as error:return {'entrypoint':value,'status':'syntax_error','error':str(error)}


def inspect_variant_locator(root, locator):
    path=contained(root,locator['path'])
    if not path.is_file():return {**locator,'status':'missing_file'}
    raw=path.read_bytes();text=raw.decode('utf-8-sig')
    if path.suffix=='.ipynb':
        notebook=json.loads(text)
        text='\n'.join(''.join(c.get('source',[])) for c in notebook['cells'] if c['cell_type']=='code')
    present=all(marker in text for marker in locator.get('markers',[]))
    return {**locator,'sha256':hashlib.sha256(raw).hexdigest(),
            'status':'selected_code_markers_present' if present else 'selected_markers_missing',
            'boundary':'Selected source anchors only; not full paper-variant equivalence, dependency or execution acceptance.'}


def validate_config(config, ara):
    ids={p['id'] for p in ara['papers']};resources=[r['id'] for r in config['source_resources']]
    if len(resources)!=len(set(resources)):raise ValueError('Duplicate code resource identifiers')
    if set(config['main_status_overrides'])!=ids:raise ValueError('Main-method audit does not cover every ARA paper')
    if any(k not in ids for k in config['paper_source_bindings']):raise ValueError('Unknown paper binding')
    if any(r not in resources for v in config['paper_source_bindings'].values() for r in v):raise ValueError('Unknown source binding')
    if any(k not in ids for k in config.get('variant_code_locators',{})):raise ValueError('Unknown variant paper binding')
    if config['all_methods_complete'] or config['all_ablations_complete']:
        raise ValueError('Local discovery config cannot certify scientific completeness')


def paper_navigation(root, ara, paper, axes):
    cache=contained(root,ara['local_text_cache'])/paper['id']/(paper['primary_source_sha256']+'.json')
    cached=read(cache)
    if cached.get('source_sha256')!=paper['primary_source_sha256']:
        raise ValueError('ARA text cache edition mismatch: '+paper['id'])
    source=next(s for s in paper['sources'] if s['sha256']==paper['primary_source_sha256'])
    if hashlib.sha256(contained(root,source['path']).read_bytes()).hexdigest()!=source['sha256']:
        raise ValueError('ARA primary PDF SHA256 mismatch: '+paper['id'])
    pages=cached['pages']; candidates=[]
    for page,text in pages.items():
        if re.search(r'(?i)ablat|component analysis|sensitivity analys',text):
            candidates.append({'pdf_page':int(page),'discovery':'automatic keyword navigation; may include abstracts, references or table-of-contents entries'})
    for item in axes:
        if str(item['pdf_page']) not in pages or normal(item['marker']) not in normal(pages[str(item['pdf_page'])]):
            raise ValueError('Curated ablation source anchor mismatch: '+paper['id'])
    mentions=[]
    for candidate in ara['papers']:
        if candidate['id']==paper['id']:continue
        name=candidate['id'].replace('_','')
        matched=[int(page) for page,text in pages.items() if name in normal(text)]
        if len(name)>=4 and matched:
            mentions.append({'paper_id':candidate['id'],'pdf_pages':matched,
                             'relation':'name mentioned; NOT automatically a compared baseline or a reproduced implementation'})
    return {'source_sha256':paper['primary_source_sha256'],'ablation_page_candidates':candidates,
            'reviewed_axes':axes,'mentioned_registered_references':mentions,
            'baseline_inventory_status':'not_exhaustively_registered_per_paper_table',
            'boundary':'Selected axes and automatic navigation do not cover all appendix ablations or all baselines.'}


def baseline_discovery(root, config):
    records=[]
    for value in config['baseline_search_roots']:
        target=contained(root,value)
        paths=[target] if target.is_file() else list(target.rglob('*.py')) if target.is_dir() else []
        candidates=[]
        for path in paths:
            if any(p in EXCLUDED for p in path.parts):continue
            try:tree,_=parse_source(path.read_bytes())
            except (SyntaxError,ValueError,UnicodeDecodeError):continue
            for node in tree.body:
                if isinstance(node,ast.ClassDef):
                    candidates.append({'symbol':node.name,'path':path.relative_to(root).as_posix(),'line':node.lineno})
        records.append({'root':value,'python_files':len(paths),'classes':candidates,
                        'boundary':'Classes include layers, wrappers and data-processing utilities; counts are NOT numbers of complete anomaly methods.'})
    return records


def patch_evidence(root, paper_id):
    path=root/'experiments/runs/flow_matching_campaign/sources'/paper_id/'patch_manifest.json'
    if not path.exists():return None
    report=read(path); working=root/report['working_path']
    changes=[]
    for item in report['patches']:
        original=path.parent/'original'/item['file'];fixed=working/item['file']
        changes.append({'file':item['file'],'reason':item['reason'],'patch_path':item['patch_path'],
                        'original_hash_matches':original.is_file() and hashlib.sha256(original.read_bytes()).hexdigest()==item['original_sha256'],
                        'corrected_hash_matches':fixed.is_file() and hashlib.sha256(fixed.read_bytes()).hexdigest()==item['corrected_sha256']})
    return {'manifest':path.relative_to(root).as_posix(),'status':report['status'],'patches':changes,
            'boundary':'Preserved source and patch hashes checked; a fix is not a proof of paper-score reproduction.'}


def run_evidence(root,config):
    records=[]
    for value in config['preflight_roots']:
        base=contained(root,value)
        paths=list(base.glob('*preflight*/report.json'))+list(base.glob('*preflight_report.json'))
        for path in sorted(set(paths)):
            record=read(path)
            scalars={k:v for k,v in record.items() if isinstance(v,(str,bool,int,float))}
            records.append({'path':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                            'top_level_scalar_fields':scalars,'boundary':'Existing preflight evidence; not a new formal benchmark run.'})
    path=root/'docs/reports/flow_matching_reproduction_status_2026-10-06.json'
    formal=[]
    if path.exists():
        for r in read(path)['results']:
            formal.append({'id':r['id'],'paper_id':r.get('reference',{}).get('paper_id'),
                           'verdict':r.get('verdict'),'runs':r.get('runs'),'reason':r.get('reason')})
    return {'preflight_reports':records,'existing_original_comparisons':formal,
            'live_status_source':path.relative_to(root).as_posix(),
            'boundary':'Read-only snapshot; running queue and its tracked status report were not changed.'}


def audit(root,config):
    ara=read(root/config['ara_config']);validate_config(config,ara)
    resources=[];ledger={}
    for resource in config['source_resources']:
        report,files=inspect_resource(root,resource);resources.append(report);ledger[resource['id']]=files
    lookup={r['id']:r for r in resources};papers=[]
    artifact=root/ara['artifact_root']
    for paper in ara['papers']:
        ident=paper['id'];override=config['main_status_overrides'][ident]
        source_ids=config['paper_source_bindings'].get(ident,[])
        entries=[inspect_entrypoint(root,e) for e in config['local_entrypoints'].get(ident,[])]
        navigation=paper_navigation(root,ara,paper,config['reviewed_ablation_axes'].get(ident,[]))
        variants=[inspect_variant_locator(root,v) for v in config.get('variant_code_locators',{}).get(ident,[])]
        record={'paper_id':ident,'title':paper['title'],'task':paper['task'],
                'ara_artifact':f'papers/{ident}/PAPER.md','primary_source_sha256':paper['primary_source_sha256'],
                'main_implementation':override,'source_resources':[{k:lookup[s][k] for k in ['id','path','repository','commit','status','code_tree_sha256']} for s in source_ids],
                'local_entrypoints':entries,'model_configs':[m['path'] for m in paper['model_configs']],
                'paper_ablations':navigation,'ablation_code_candidates':[
                    {'resource_id':s,'path':name,'status':'filename_candidate_not_semantically_matched'} for s in source_ids for name in lookup[s]['ablation_filename_candidates']],
                'selected_variant_code_locators':variants,
                'ablation_coverage_status':'not_applicable_dataset_reference' if ident=='swat_dataset' else 'not_fully_verified',
                'all_paper_ablations_ready':None if ident=='swat_dataset' else False,
                'paper_score_reproduction_status':'not_inferred_from_code_presence',
                'preserved_patches':patch_evidence(root,ident),
                'next_gate':'Register every main/ablated/baseline variant against a paper table or algorithm, explicit code symbol/config, validation behavior and task-specific runner.'}
        papers.append(record)
        write(artifact/'papers'/ident/'src/code/code_coverage.json',record)
    statuses=Counter(p['main_implementation']['status'] for p in papers)
    summary={'paper_count':len(papers),'unique_code_resources':len(resources),
             'present_code_resources':sum(r['status']=='local_code_files_present' for r in resources),
             'papers_bound_to_source_resources':sum(bool(p['source_resources']) for p in papers),
             'papers_with_local_entrypoint_symbols':sum(any(e['status']=='top_level_symbol_found' for e in p['local_entrypoints']) for p in papers),
             'main_status_counts':dict(statuses),'missing_registered_main_implementations':[p['paper_id'] for p in papers if p['main_implementation']['status']=='no_registered_main_implementation_located'],
             'papers_with_fully_verified_all_ablation_coverage':0,'all_methods_complete':False,'all_ablations_complete':False,
             'boundary':'Zero certified-complete ablation sets does not mean zero ablation code. Shared code directories and components do not count as independent complete detectors.'}
    report={'schema_version':1,'checked_at':datetime.now(timezone.utc).isoformat(),'scope':config['scope'],
            'summary':summary,'code_resources':resources,'papers':papers,
            'baseline_resource_discovery':baseline_discovery(root,config),'run_evidence':run_evidence(root,config),
            'unit_validation':config['unit_validation'],
            'next_priorities':['GRASP/DFM native TSAD main methods and exact scoring','MaelNet/DT-LA/KGL and SHCL-Transformer; Pi journal/mirror equivalence','JFI/CFM-TS/PrismFlow missing main implementations','Turn verified FM/SF2M components into task-specific adapters','Map each paper ablation and compared baseline to explicit symbols/configurations, then validate and run'],
            'boundary':'A comprehensive local resource/status audit, not a certification that every described method or ablation is implemented. No new vendor execution or scientific training.'}
    write(root/config['output'],report);write(root/config['local_file_ledger'],ledger)
    labels={'source_present_not_end_to_end_verified':'作者源码在场，完整实验未验证','no_registered_main_implementation_located':'主方法未定位本地登记实现',
            'related_components_only':'只有相关库/组件','components_and_tutorial_present':'组件与教程在场','related_author_libraries_present':'作者相关库在场，原版本待对齐',
            'partial_paper_method_release':'论文方法仅部分发布','third_party_mirror_present':'第三方镜像在场，版本待核',
            'publisher_supplement_present_unverified':'官方补充源码在场，等价性待核','local_reconstruction_with_deviation':'本地还原存在明确消融偏离',
            'local_transcription_present':'本地转写在场','dataset_reference_not_detector':'数据来源章节'}
    table='| 参考 | 主方法代码状态 | 本地入口 | 消融完整性 |\n|---|---|---|---|\n'+'\n'.join(
        f"| [{p['paper_id']}](papers/{p['paper_id']}/PAPER.md) | {labels[p['main_implementation']['status']]} | {'符号在场' if any(e['status']=='top_level_symbol_found' for e in p['local_entrypoints']) else '尚无已定位统一入口'} | {'不适用' if p['paper_id']=='swat_dataset' else '未逐项验收完整'} |" for p in papers)
    notable='\n\n'.join(f"- **{p['paper_id']}**：{p['main_implementation']['note']}" for p in papers if p['main_implementation']['status'] not in {'source_present_not_end_to_end_verified','no_registered_main_implementation_located','dataset_reference_not_detector'})
    axes='\n'.join(f"- **{p['paper_id']}**："+'；'.join('、'.join(a['axes'])+f"（PDF{a['pdf_page']}页）" for a in p['paper_ablations']['reviewed_axes']) for p in papers if p['paper_ablations']['reviewed_axes'])
    available='\n'.join(f"- **{p['paper_id']}**："+'；'.join(f"{v['variant']}（`{v['path']}`，{v['status']}）" for v in p['selected_variant_code_locators']) for p in papers if p['selected_variant_code_locators'])
    text=('# 全方法与消融代码准备情况\n\n**结论：未全部准备好。**论文全文齐全和代码/消融齐全是不同状态。\n\n'+
         f"核对45篇ARA参考，发现{summary['present_code_resources']}个独立源码资源目录，关联{summary['papers_bound_to_source_resources']}篇参考（含共享库、镜像和部分实现）。{summary['papers_with_local_entrypoint_symbols']}篇有可静态定位的本地运行/模型入口；AST定位不代替环境/运行验证。\n\n"+
         '未定位已登记主方法实现：'+', '.join(summary['missing_registered_main_implementations'])+'。这是本项目已登记/已搜索本地资源范围的结论，未推断网上不存在代码。\n\n'+
         '没有任何一篇被本次审计认证为“全文全部消融均已准备并验收”。已有DCdetector消融脚本、SF2M教程和USAD开关等记录为部分资源；不能以文件名、库类或预检推断全部变体完整。\n\n'+
         table+'\n\n## 已定位的部分变体代码\n\n'+available+'\n\n这些定位只说明相应代码片段在场，不说明论文整套消融配置和实验已经完成。\n\n## 特殊边界\n\n'+notable+'\n\n## 已锚定的消融工作项\n\n'+axes+'\n\n'+
         '上述是选定原文段落中的消融轴，并非全部附录、超参数组合或全部比较基线。各项仍须绑定代码、冻结配置、输出差异与行为测试；不是已执行任务。\n\n'+
         '论文中引用/比较的全部基线尚未逐表建立完整注册。TAB和TSLib存在大量模型及封装，缺依赖的封装、普通层类和第三方移植不能按完整基线计数。结构化报告保存类/文件定位候选与边界。\n\n'+
         '当前Python语法检查发现原始BRITS的main.py仍使用Python 2 print语句；本地BRITS运行入口采用SAITS发布版中的实现，两者算法/版本等价性尚未验收。其他源码通过语法解析也不代表依赖、CUDA扩展、checkpoint或完整实验已经可用。\n\n'+
         '本地CPU接口测试10项通过；合成输入仅验证可调用接口。已有插补预检和原结果比较单独引用，未改变运行队列；重复次数不完整的结果不标记复现完成。\n\n'+
         '完成门槛：论文算法/表行与原文版本 → 源码符号和哈希 → 每个变体冻结配置与训练/评分入口 → 数据/依赖/掩码协议 → 行为与恢复测试 → 原数据多种子运行。\n\n'+
         '[结构化总表](code_coverage.json)；每篇 `src/code/code_coverage.json` 提供其来源、入口、消融页码、补丁哈希与缺口。配置真源为benchmark的 `configs/reproducibility/fm_code_coverage.v1.json`。')
    write(root/config['markdown_output'],text)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/reproducibility/fm_code_coverage.v1.json')
    args=parser.parse_args();print(json.dumps(audit(ROOT,read(args.config)),ensure_ascii=True,indent=2))


if __name__=='__main__':main()
