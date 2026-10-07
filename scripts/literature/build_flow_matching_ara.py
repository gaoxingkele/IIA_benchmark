"""Build source-anchored ARA engineering files, without claiming experiment success.

Configuration owns paths and scientific annotations. Full PDF extraction is local
and ignored; only summaries, page indexes and provenance enter the artifact.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import fitz
import yaml

from scripts.literature.import_flow_matching_archives import contained, normal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from iia_benchmark.config.storage import project_relative_path


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    path.write_text(value.rstrip()+'\n', encoding='utf-8', newline='\n')


def inspect_source(root, source):
    path = contained(root, source['path'])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != source['sha256']:
        raise ValueError('Source SHA256 mismatch: '+source['path'])
    with fitz.open(path) as doc:
        if doc.page_count != source['pages']:
            raise ValueError('Source page count mismatch')
        selected = source['selected_pdf_pages']
        if not selected or len(set(selected)) != len(selected) or any(p < 1 or p > doc.page_count for p in selected):
            raise ValueError('Invalid source page selection')
        pages = {p:doc[p-1].get_text() for p in selected}
    return pages


def validate_annotations(paper, pages_by_hash):
    primary = paper['primary_source_sha256']
    if primary not in pages_by_hash:
        raise ValueError('Unregistered primary source')
    if paper['scientific_reproduction_status'] != 'not_established_by_this_artifact':
        raise ValueError('Source engineering cannot declare a reproduced result')
    for anchor in paper['anchors']:
        pages = pages_by_hash[anchor.get('source_sha256',primary)]
        if anchor['pdf_page'] not in pages or normal(anchor['marker']) not in normal(pages[anchor['pdf_page']]):
            raise ValueError('Unresolved page/marker anchor: '+paper['id']+'/'+anchor['id'])
    for result in paper['reported_results']:
        if result.get('status') != 'author_reported_not_locally_reproduced':
            raise ValueError('Reported table cannot declare local performance')
        if result['pdf_page'] not in pages_by_hash[result.get('source_sha256',primary)]:
            raise ValueError('Result outside selected paper pages')


def code_mapping(root, ident):
    matches=[]
    manifest=root/'papers/literature/flow_matching/author_code_manifest.json'
    if manifest.exists():
        matches += [r for r in read(manifest)['records'] if r.get('paper_id')==ident]
    config=root/'configs/reproducibility/mtsad_protocol_sources.v1.json'
    alias={'timesnet':'timesnet_tslib','moment':'moment_research','flow_matching_guide':'flow_matching_library'}
    if config.exists():
        matches += [r for r in read(config)['papers'] if r['id']==alias.get(ident,ident)]
    if ident=='pi_transformer':
        matches+=read(root/'configs/acquisition/fm_project_missing_papers.v1.json')['repositories']
    return matches


def data_mapping(root, paper):
    registries=[]
    for name in ['flow_matching_fm_data_sources.json','flow_matching_comparison_data_sources.json',
                 'flow_matching_foundation_data_sources.json','fm_project_missing_datasets.v1.json',
                 'fm_ara_data_gaps.v1.json']:
        path=root/'configs/acquisition'/name
        if not path.exists():
            continue
        items=[s for s in read(path).get('sources',[]) if paper['id'] in {normal(i) for i in s.get('paper_ids',[])} or normal(paper['id']) in {normal(i) for i in s.get('paper_ids',[])}]
        if items:
            registries.append({'registry':path.relative_to(root).as_posix(),'registered_items':len(items),
                              'items':[{'id':s['id'],'path':s.get('path'),'access':s.get('access'),
                                        'local_payload_present':bool(s.get('path') and (root/project_relative_path(root,s['path'])).is_file())} for s in items]})
    classic=[]
    for ds in paper['original_datasets']:
        key=ds.lower().replace('swat','swat')
        path=root/'configs/datasets'/('mtsad_'+key+'.json')
        if path.exists(): classic.append(path.relative_to(root).as_posix())
    return {'paper_id':paper['id'],'original_datasets':paper['original_datasets'],
            'classic_dataset_configs':classic,'registries':registries,
            'boundary':'Presence is a local path check, not a new dataset checksum, split audit or exact preprocessing match. Unlisted source datasets remain protocol/data gaps.'}


def build_paper(root, output, cache, paper):
    ident=paper['id']; target=output/'papers'/ident
    pages={s['sha256']:inspect_source(root,s) for s in paper['sources']}
    validate_annotations(paper,pages)
    for digest, content in pages.items():
        write(cache/ident/(digest+'.json'),{'source_sha256':digest,'pages':{str(k):v for k,v in content.items()}})
    primary=pages[paper['primary_source_sha256']]
    sources=[{**s,'integrity_status':'fresh_sha256_and_pdf_parse_verified',
              'indexed_selected_pages':len(pages[s['sha256']]),'text_coverage':'selected pages extracted; semantic review is limited to listed anchors and annotations'} for s in paper['sources']]
    indexes=[{'pdf_page':p,'characters':len(t),'has_table_marker':'table' in t.lower(),
              'has_figure_marker':'figure' in t.lower() or 'fig.' in t.lower(),
              'review':'automatic navigation index; table/figure presence is not semantic verification'} for p,t in primary.items()]
    source_report={'paper_id':ident,'primary_source_sha256':paper['primary_source_sha256'],
                   'sources':sources,'anchors':paper['anchors'],'page_index':indexes,
                   'scientific_reproduction_status':paper['scientific_reproduction_status']}
    write(target/'metadata.json',paper)
    write(target/'evidence/source/source_manifest.json',source_report)
    write(target/'evidence/tables/reported_results.json',{'paper_id':ident,'results':paper['reported_results'],
          'status':'paper_claims_only','caveat':paper.get('table_caveat'),
          'boundary':'No score here was produced by a local experiment. Rounding, uncertainty definition, PA and threshold partition must match before comparing.'})
    write(target/'evidence/runs/local_validation.json',{
        'E01':{'status':'passed','meaning':'PDF hash, structure and selected pages verified'},
        'E02':{'status':'documentary_anchors_resolved','meaning':'Marker/page navigation verified; scientific content is a source-reported description'},
        'E03':{'status':'not_executed_by_this_iteration','meaning':'Original author protocol reproduction; inspect existing campaign evidence separately'},
        'E04':{'status':'planned','meaning':'Task-compatible common protocol and transfer experiment; no benchmark result'},
        'campaign_status_at_registry_creation':paper['campaign_status']})
    mapping={'paper_id':ident,'model_configs':paper['model_configs'],'code_url':paper.get('code_url'),
             'pinned_sources':code_mapping(root,ident),'boundary':'Pinned sources and callable entrypoints are separate evidence; acquisition is not reproduction.'}
    write(target/'src/code/implementation_mapping.json',mapping)
    write(target/'src/environment.json',paper.get('environment_reference',{
          'status':'requires_author_environment_lock','boundary':'No new environment was created during source engineering.'}))
    data=data_mapping(root,paper)
    write(target/'src/configs/dataset_mapping.json',data)
    heading='# '+paper['title']+'\n\n'
    write(target/'PAPER.md',heading+f"研究任务：`{paper['task']}`；方法族：`{paper['family']}`。\n\n"+
          '本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。\n\n'+
          '四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。\n\n'+
          '主要复现问题：'+paper['reproduction_gap'])
    write(target/'logic/problem.md',heading+'原任务：'+paper['task']+'。\n\n'+
          '在流匹配研究中的用途：原生检测方法可进入检测对比；插补/预测/生成方法先复现原任务，再通过冻结的评分适配进入新轨道；基础理论和数据来源只提供设计与来源证据。\n\n'+
          '原文数据集：'+('；'.join(paper['original_datasets']) or '尚未结构化逐项核实；从页索引定位实验章节后登记，不能视为没有实验数据。'))
    write(target/'logic/concepts.md',heading+f"方法族：`{paper['family']}`。任务：`{paper['task']}`。\n\n"+
          '本文件使用四种证据强度：原文描述、原文报告数值、本地素材/接口验证、本地完整实验。前三者不能代替第四种。')
    write(target/'logic/solution/method.md',heading+paper['method_summary']+'\n\n'+
          '原文导航：\n\n'+'\n'.join(f"- {a['id']}: PDF 第 {a['pdf_page']} 页，检索标记 `{a['marker']}`；{a['review']}。" for a in paper['anchors'])+'\n\n'+
          '工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。')
    write(target/'logic/solution/constraints.md',heading+paper['reproduction_gap']+'\n\n'+
          '冻结要求：作者版本与代码提交、数据实体/划分、训练专属归一化、掩码/窗口及回填、种子和预算、积分器/步数、分数聚合、阈值来源、点调整、指标尺度与不确定性。\n\n'+
          '可直接运行的本地模型配置及当前状态见 `src/code/implementation_mapping.json`；没有 entrypoint 的配置表示待适配，不能宣称方法已实现。')
    write(target/'logic/related_work.md',heading+
          f"在集合中归入 `{paper['family']}`，原任务为 `{paper['task']}`。其对比资格由任务、数据版本和评分协议决定。\n\n"+
          '完整集合分类和竞争方法见上级 README；共同实验设计引用仓库 `configs/experiments/fm_mtsad_comparison.v1.json`。此集合覆盖已注册参考，未递归声称所有参考文献均已审读。')
    claims=[]
    statements=[('C01','所登记全文对应独立校验的来源及指定页范围。','来源版本、SHA256 和页范围匹配。','文件哈希改变、标题/章节不符、页码越界。','E01'),
                ('C02',paper['method_summary'],'这是原文机制描述，未声称完成理论证明或数值复现。','锚点不能定位对应机制，或任务/机制被版本核对推翻。','E02'),
                ('C03',('原文登记 '+str(len(paper['reported_results']))+' 条结果，具体数据集、值和页码见 evidence/tables/reported_results.json；其本地等效性尚待原协议重复实验。' if paper['reported_results'] else '该文数值尚未逐项转录；不能给出未核实的论文指标或本地等效性结论。')+' '+paper['reproduction_gap'],'数据实体、掩码、预算、阈值、PA、聚合及不确定性定义相同。','多种子差异超出预先设定的等效边界，或原数据/协议无法对齐。','E02, E03'),
                ('C04','对可适配检测的方法，在共同协议下的相对性能是待检验问题。','先通过任务适配门禁；基础/数据参考只参与设计，不进入检测排行。','严格协议下性能下降或排序变化，或评分使用不可用的测试标签。','E04')]
    for cid,statement,conditions,falsification,proof in statements:
        claims.append(f'## {cid}\n\n**Statement**: {statement}\n\n**Conditions**: {conditions}\n\n**Falsification criteria**: {falsification}\n\n**Proof**: {proof}\n\n**Status**: '+('source_verified' if cid=='C01' else 'source_reported' if cid=='C02' else 'pending_experiment'))
    write(target/'logic/claims.md',heading+'\n\n'.join(claims))
    experiment=[]
    specs=[('E01','C01','本地原文完整性和章节核验','读取登记 PDF、重新计算 SHA256、解析指定范围；保存来源清单。','与原始登记匹配，书籍仅按章节范围建立证据。','evidence/source/source_manifest.json'),
           ('E02','C02, C03','文献机制与导航依据','核验配置中标记确在对应页；人工注释范围与全文自动索引区分；有数字的表格单独标记转录来源。','锚点可解析，不推断原文性能成立。','evidence/source/source_manifest.json'),
           ('E03','C03','原作者协议复现，状态待执行/另行登记','锁定代码、数据划分和表格版本；先接口检查，再按原预算多种子运行；以预先冻结的等效区间比较。','取得实际运行日志、预测及配对差异；当前无本轮运行结果。','evidence/runs/local_validation.json'),
           ('E04','C04','共同异常检测协议与工业迁移，状态 planned','依 comparison config 分原作者、TAB、严格验证、工业迁移四轨；报告 AUROC/AP、无PA点F1、实体宏平均、误报/延迟及配对不确定性。','检验性能差异来自评分协议、数据/训练预算还是模型机制；适配不合格者退出排行榜。','evidence/runs/local_validation.json')]
    for eid,cid,setup,procedure,outcome,evidence in specs:
        experiment.append(f'## {eid}\n\n**Verifies**: {cid}\n\n**Setup**: {setup}\n\n**Procedure**: {procedure}\n\n**Expected outcome**: {outcome}\n\n**Evidence**: `{evidence}`')
    write(target/'logic/experiments.md',heading+'\n\n'.join(experiment))
    write(target/'evidence/README.md',heading+
          '来源与页码：`evidence/source/source_manifest.json`；论文数值：`evidence/tables/reported_results.json`；本地状态：`evidence/runs/local_validation.json`。\n\n'+
          '完整全文提取留在本地 ignored cache；Git 只保存摘要、页索引、哈希与选定数值。章节来源不会把整本书的其余章节计作文献覆盖。')
    write(target/'evidence/source/source_overview.md',heading+'\n\n'.join(
          f"来源 `{s['path']}`\n\nSHA256 `{s['sha256']}`；整文件 {s['pages']} 页，本论文指定范围 {s['selected_pdf_pages'][0]}–{s['selected_pdf_pages'][-1]}。" for s in sources))
    rows=paper['reported_results']
    table='| 数据集 | 指标 | 论文值 | PDF页 | 表/方法 |\n|---|---|---:|---:|---|\n'+ '\n'.join(
          f"| {r['dataset']} | {r['metric']} | {r.get('value',r.get('mean'))} | {r['pdf_page']} | {r.get('table','')} / {r.get('method_row',ident)} |" for r in rows) if rows else '原文数值尚未逐项转录。已有全文不代表表格数字已经核实；按页索引定位后加入冻结参考。'
    write(target/'evidence/tables/README.md',heading+table+'\n\n'+(paper.get('table_caveat') or '')+'\n\n状态：全部为作者报告值；原单位、误差定义和协议见 JSON。不得作为本地排行榜结果。')
    write(target/'evidence/figures/README.md',heading+'图示导航见 source_manifest 的 has_figure_marker；该字段为自动文本检索。未导出的图和未人工审读的图不宣称已核实。')
    write(target/'src/environment.md',heading+
          '复现环境由已有 campaign 和模型配置确定，按作者锁定的 Python/PyTorch/CUDA 建独立环境，避免改动运行中的队列。\n\n'+
          'FM 库、S4 扩展和基础预训练模型需要各自依赖与权重；硬件/精度/推断成本与训练预算在结果中报告。素材工程本轮不启动训练。')
    write(target/'src/code/README.md',heading+paper['reproduction_gap']+'\n\n'+
          '代码来源、提交、模型 entrypoint 和状态见 implementation_mapping.json。原始代码保留；后续修正存独立副本、补丁和行为检验，不能覆盖作者快照。')
    write(target/'src/configs/README.md',heading+'路径和实验参数以仓库 configs 为真源。dataset_mapping.json 只记录来源与存在性，不能作为精确 split-ready 声明。\n\n'+
          '原始文件只读；预处理写衍生目录并记录输入哈希、输出哈希与实体分组；数据物理位置沿用 F 盘存储配置。')
    tree={'schema_version':1,'paper_id':ident,'nodes':[
          {'id':'source','support_level':'source_integrity_verified','evidence':['C01','E01'],'next':'mechanism'},
          {'id':'mechanism','support_level':'documentary_only','evidence':['C02','E02'],'next':'original_reproduction'},
          {'id':'original_reproduction','support_level':'pending_actual_runs','evidence':['C03','E03'],'next':'common_protocol'},
          {'id':'common_protocol','support_level':'planned_task_gated','evidence':['C04','E04'],'next':None}]}
    write(target/'trace/exploration_tree.yaml',yaml.safe_dump(tree,allow_unicode=True,sort_keys=False))
    return {'id':ident,'title':paper['title'],'task':paper['task'],'family':paper['family'],
            'artifact':target.relative_to(output).as_posix(),'source_files':len(sources),
            'reported_rows':len(rows),'formal_reproduction':'not_established_by_this_artifact'}


def build(root, config):
    output=contained(root,config['artifact_root']);cache=contained(root,config['local_text_cache'])
    ids=[p['id'] for p in config['papers']]
    if len(set(ids))!=len(ids) or len(ids)!=config['paper_count']:
        raise ValueError('Paper ids/count mismatch')
    records=[build_paper(root,output,cache,p) for p in config['papers']]
    write(output/'index.json',{'schema_version':1,'paper_count':len(records),'papers':records,
          'scope':config['scope'],'source_configuration':'configs/reproducibility/flow_matching_ara.v1.json',
          'boundary':'Source engineering validated; no new scientific experiment results.'})
    table='| 参考 | 原任务 | 方法族 | ARA入口 |\n|---|---|---|---|\n'+'\n'.join(
          f"| {r['id']} | {r['task']} | {r['family']} | [工程](papers/{r['id']}/PAPER.md) |" for r in records)
    write(output/'README.md','# 流匹配参考文献 ARA 工程\n\n'+
          f"本集合含 {len(records)} 篇独立参考论文，按逻辑、证据、实现及追溯四层建档。每篇有来源哈希/版本/页码、方法与约束、可证伪主张、实验设计、代码/数据映射及明确的实验状态。\n\n"+
          '配置真源：benchmark 仓库 configs/reproducibility/flow_matching_ara.v1.json。集合覆盖已注册 FM 及基础、八个指定竞争方法、后续版本、用户补充的数据/方法来源、TAB 与 Anomaly Transformer；并非递归的全部引用网络。\n\n'+
          '本轮只做素材核验和工程建档。原作者结果与本地实验结果分开；当前完整复现与统一异常检测对比仍需实际运行证据。未逐项转录的表格有明确状态，不填造数字。\n\n'+
          '[补件核对](archive_reconciliation.md) · [方法族映射](method_families.md) · [比较与失配分析](comparison_plan.md) · [结构化索引](index.json)\n\n'+table+'\n\n'+
          '更新与验证（从 benchmark 根运行）：\n\n```powershell\npython -m scripts.literature.import_flow_matching_archives\npython -m scripts.literature.build_flow_matching_ara\npython -m scripts.literature.verify_flow_matching_ara\n```')
    write(output/'comparison_plan.md','# 比较协议与性能失配分析\n\n'+
          '实验设计以 configs/experiments/fm_mtsad_comparison.v1.json 为准。四轨分别是作者原协议、固定提交 TAB、严格验证集校准、工业迁移；任何测试标签选择阈值的结果只能留在原协议复现轨。\n\n'+
          '先锁定目标：原生 TSAD 的 F1/AUROC/AP；插补的 MAE/RMSE/CRPS；预测/生成的原指标。基础库和图像方法只有经过显式任务适配才能进入时序排行。\n\n'+
          '性能不匹配的排查顺序：\n\n'+
          '1. 数据版本、实体集合、列选择、缺失率、标签和训练污染；特别检查 CrossAD 首特征轨、GRASP mTSBench 与 JFI TEP 仿真子集。\n'+
          '2. 归一化拟合划分、插值是否跨 test、窗口长度/步长、重叠回填、标签点/段定义。\n'+
          '3. 阈值是否用测试分布或测试标签、best-test 搜索、point adjustment、宏/微平均；同时报告未调整 AUROC/AP。\n'+
          '4. 作者代码与论文版本、超参数选择、种子、训练/推断预算、checkpoint 和预训练重叠。\n'+
          '5. FM 路径/耦合、图谱权重、条件先验、速度失配时间积分、solver/NFE、观测投影、分数方差。\n\n'+
          '因果分析采用单因素消融及配对种子/实体 bootstrap，不在测试集上选最佳方案。先冻结等效界值和容差，再比较报告的均值及不确定性；作者未给误差的结果不能凭四舍五入即宣称复现。\n\n'+
          '未优于基线时先报告不确定性与成本，再区分协议失配、实现缺陷、优化失败与方法能力限制。人工修正必须留下独立补丁、触发条件和有意义的行为验证。')
    return {'paper_count':len(records),'source_editions':sum(r['source_files'] for r in records),
            'reported_rows':sum(r['reported_rows'] for r in records),'task_counts':dict(Counter(r['task'] for r in records))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/reproducibility/flow_matching_ara.v1.json')
    args=parser.parse_args();print(json.dumps(build(ROOT,read(args.config)),ensure_ascii=True))


if __name__=='__main__': main()
