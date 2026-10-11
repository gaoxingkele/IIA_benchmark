"""Render an audited metric CSV without changing measurements or repeat counts."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

from scripts.flow_matching.publish_result_listing_v7 import ROOT, validate_rows


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(rows, headers, version, source_digest):
    numeric = sum(row['mean'] != '' for row in rows)
    undefined = sum('evaluations_completed_affiliation_F_undefined' in row['comparison'] for row in rows)
    payload = json.dumps(rows, ensure_ascii=False).replace('<', '\\u003c')
    columns = json.dumps(headers, ensure_ascii=False).replace('<', '\\u003c')
    heading = html.escape(version)
    summary = (f'{len(rows):,} 个指标行；{numeric:,} 行有数值，{len(rows)-numeric:,} 行暂无数值。'
               f'其中 {undefined} 行已明确记录为评估完成、指标无定义。指标行数不等于实验数量。')
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>算法—数据集指标 {heading}</title><style>'
            'body{font:14px system-ui;margin:24px;color:#18242d}p{line-height:1.7}'
            'input{padding:10px;width:min(680px,90%)}button{padding:8px;margin:8px 8px 8px 0}'
            '.scroll{overflow:auto;max-height:70vh}table{border-collapse:collapse}'
            'th,td{border:1px solid #d7dfe6;padding:7px;white-space:nowrap}'
            'th{position:sticky;top:0;background:#edf3fa}small{display:block;margin:8px 0;overflow-wrap:anywhere}'
            '</style></head><body>'
            f'<h1>算法—数据集指标 {heading}</h1><p id="summary">{summary}</p>'
            '<p>尚未完成全部原始实验。论文、严格验证集阈值、测试标签选择和其他协议分别列示；'
            '训练中间进度不计作最终性能。空值保留，不替换为 0。请结合 comparison、uncertainty_scope、'
            'author_equivalence_certified 和 strict_TAB_result 阅读每一行。</p>'
            '<p><a href="all_algorithm_dataset_metrics.csv">下载完整 CSV</a> · '
            '<a href="all_metrics.md">完整 Markdown 表</a></p>'
            '<label for="filter">筛选算法、数据集、协议或指标</label><br>'
            '<input id="filter" placeholder="例如 PRONTO_DAY4 或 validation_only">'
            '<small id="count"></small><button id="previous">上一页</button><button id="next">下一页</button>'
            '<div class="scroll"><table><thead><tr>'
            + ''.join('<th>'+html.escape(key)+'</th>' for key in headers)
            + '</tr></thead><tbody id="body"></tbody></table></div>'
            f'<small>源 CSV SHA256：{source_digest}</small>'
            '<script id="metric-data" type="application/json">'+payload+'</script>'
            '<script id="metric-columns" type="application/json">'+columns+'</script>'
            '<script>const rows=JSON.parse(document.getElementById("metric-data").textContent);'
            'const cols=JSON.parse(document.getElementById("metric-columns").textContent);'
            'const input=document.getElementById("filter"),body=document.getElementById("body");'
            'let page=0;const size=100;'
            'function render(){const q=input.value.toLowerCase();'
            'const kept=rows.filter(r=>Object.values(r).join(" ").toLowerCase().includes(q));'
            'const pages=Math.max(1,Math.ceil(kept.length/size));page=Math.min(page,pages-1);'
            'body.replaceChildren();const frag=document.createDocumentFragment();'
            'for(const r of kept.slice(page*size,(page+1)*size)){const tr=document.createElement("tr");'
            'for(const c of cols){const td=document.createElement("td");'
            'td.textContent=r[c]===null||r[c]===undefined||r[c]===""?"—":String(r[c]);tr.append(td)}'
            'frag.append(tr)}body.append(frag);'
            'document.getElementById("count").textContent="匹配 "+kept.length+" / "+rows.length+'
            '" 个指标行；第 "+(page+1)+" / "+pages+" 页；每页最多 "+size+" 行";'
            'document.getElementById("previous").disabled=page===0;'
            'document.getElementById("next").disabled=page>=pages-1}'
            'input.addEventListener("input",()=>{page=0;render()});'
            'document.getElementById("previous").addEventListener("click",()=>{page--;render()});'
            'document.getElementById("next").addEventListener("click",()=>{page++;render()});'
            'render();</script></body></html>')


def publish(publication):
    base = (ROOT/publication).resolve()
    if not base.is_relative_to(ROOT.resolve()):
        raise ValueError('Presentation must stay within the current research workspace')
    csv_path = base/'all_algorithm_dataset_metrics.csv'
    before = csv_path.read_bytes()
    receipt = json.loads((base/'publication_validation.json').read_text(encoding='utf-8'))
    independent = json.loads((base/'independent_publication_review.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256(before).hexdigest()
    if (receipt['outputs']['all_algorithm_dataset_metrics.csv'] != digest
            or independent['reviewed_csv_sha256'] != digest or not independent['passed']):
        raise ValueError('Exact audited CSV and independent publication review required')
    with csv_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        headers, rows = list(reader.fieldnames), list(reader)
    validate_rows(rows)
    numeric = sum(row['mean'] != '' for row in rows)
    if len(rows) != receipt['aggregate_metric_rows'] or len(headers) != receipt['fields'] or numeric != receipt['measured_metric_rows']:
        raise ValueError('Source count and measurement scope differ')
    version = base.name.rsplit('_', 1)[-1]
    summary = (f'{len(rows):,} 个指标行，{numeric:,} 行有数值，{len(rows)-numeric:,} 行暂无数值。'
               '无数值行包括完成评估后指标无定义的情况，不等于未运行的实验数。全部原始实验仍未完成。')
    def cell(value):
        return (value if value != '' else '—').replace('|', '\\|').replace('\n', ' ')
    markdown = '# 算法—数据集指标 '+version+'\n\n'+summary+'\n\n'
    markdown += '| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'
    markdown += '\n'.join('| '+' | '.join(cell(row[key]) for key in headers)+' |' for row in rows)+'\n'
    page = render(rows, headers, version, digest)
    import re
    embedded = re.search(r'<script id="metric-data" type="application/json">(.*?)</script>', page, re.S)
    if json.loads(embedded.group(1)) != rows or csv_path.read_bytes() != before:
        raise ValueError('Every cell must be unchanged in presentation and source')
    paths = [base/name for name in ('results.html', 'all_metrics.md', 'presentation_validation.json')]
    if any(path.exists() for path in paths):
        raise ValueError('Preserve existing published presentation files')
    for path, content in zip(paths[:2], [page, markdown]):
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(content)
    review = dict(captured_utc=datetime.now(timezone.utc).isoformat(), source_csv_sha256=digest,
                  source_publication_receipt_sha256=sha(base/'publication_validation.json'),
                  source_independent_review_sha256=sha(base/'independent_publication_review.json'),
                  renderer_sha256=sha(Path(__file__)), rows=len(rows), fields=len(headers), numeric_rows=numeric,
                  without_numeric_rows=len(rows)-numeric, embedded_all_cells_equal_source_CSV=True,
                  original_CSV_bytes_unchanged=True, new_numeric_results=0, page_size=100,
                  old_published_presentations_preserved=True, all_original_experiments_complete=False,
                  outputs={path.name:sha(path) for path in paths[:2]})
    with paths[2].open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(review, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key:review[key] for key in ('rows','fields','numeric_rows','without_numeric_rows','new_numeric_results')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication', required=True)
    publish(parser.parse_args().publication)
