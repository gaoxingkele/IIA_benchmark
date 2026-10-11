"""Append one fully replayed original Pi seed while preserving every prior slot."""
import argparse
import csv
from datetime import datetime, timezone
import html
import io
import json

from scripts.flow_matching.audit_pi_original_recovery_completion_v1 import ROOT, read, digest
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json
from scripts.flow_matching.publish_brits_two_seed_listing_v27 import legacy


ALGORITHM = 'pi_transformer_historical_mirror/source_runtime/full'
METRICS = [('author_threshold_PA', k) for k in ('precision','recall','f1')] + [
    ('same_author_threshold_no_PA', k) for k in ('precision','recall','f1','auroc','average_precision')]


def check_delta(before, after):
    if after[:len(before)] != before or len(after) != len(before)+8:
        raise ValueError('Every prior metric row and capture time must remain unchanged')
    added = after[len(before):]
    if [(r['protocol'],r['metric']) for r in added] != METRICS:
        raise ValueError('Exactly the eight separate original PA and pointwise metrics are required')
    if any(r['algorithm'] != ALGORITHM or r['dataset'] != 'PSM' or r['n'] != '1'
           or r['required_repeats'] != '6' or r['strict_TAB_result'] != 'False'
           or r['author_equivalence_certified'] != 'False'
           or any(r[k] for k in ('std','se','ci95_low','ci95_high')) for r in added):
        raise ValueError('One original seed cannot certify uncertainty, full six-seed coverage or TAB')
    if any(r['algorithm'] == ALGORITHM and r['dataset'] == 'PSM' for r in before):
        raise ValueError('Do not count a canonical original seed twice')
    return added


def interactive_table(rows, headers):
    payload = json.dumps(rows,ensure_ascii=False).replace('<','\\u003c')
    columns = json.dumps(headers)
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>实验技术指标 v42</title>'
      '<style>body{font:14px system-ui;margin:24px}input{padding:8px;width:480px}table{border-collapse:collapse}'
      'td,th{padding:7px;border:1px solid #ddd;white-space:nowrap}th{position:sticky;top:0;background:#edf3fa}'
      '.scroll{overflow:auto;max-height:78vh}small{display:block;margin:12px 0}</style>'
      '<h1>算法—数据集—技术指标 v42</h1><p>8,626 项；2,128 项有实测值，6,498 项待完成。</p>'
      '<p>新增 Pi 原始 PSM seed42 的 8 项指标。测试集校准和 PA 单列；不属于严格 TAB 结果。</p>'
      '<input id="filter" placeholder="筛选算法、数据集、协议或指标"><small id="count"></small>'
      '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in headers)
      +'</tr></thead><tbody id="body"></tbody></table></div><script>const rows='+payload+', cols='+columns+';'
      'const input=document.getElementById("filter"),body=document.getElementById("body");'
      'function render(){const q=input.value.toLowerCase();const kept=rows.filter(r=>Object.values(r).join(" ").toLowerCase().includes(q));'
      'body.replaceChildren();const frag=document.createDocumentFragment();for(const r of kept){const tr=document.createElement("tr");'
      'for(const c of cols){const td=document.createElement("td");td.textContent=r[c]||"—";tr.append(td)}frag.append(tr)}'
      'body.append(frag);document.getElementById("count").textContent="显示 "+kept.length+" / "+rows.length+" 项；共 "+cols.length+" 列"}'
      'input.addEventListener("input",render);render();</script></html>')


def publish(config_path):
    config = read(config_path)
    for item in config['source_receipts']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Frozen publication evidence changed: '+item['path'])
    proof = read(config['full_replay_proof'])
    review_receipt = read(config['review_receipt'])
    if (not proof['passed'] or not proof['full_original_model_checkpoint_reinference_completed']
            or not proof['all_fifteen_full_saved_and_replay_streams_independently_reread']
            or proof['original_training_seed_slots'] != 1 or proof['seed'] != 42
            or proof['branch'] != 'source_runtime' or proof['dataset'] != 'PSM'
            or proof['original_required_training_seeds'] != [42,1103,1104,1105,1106,1107]
            or proof['additional_training_seed_slots'] != 0 or proof['strict_TAB_result']
            or proof['author_equivalence_certified'] or review_receipt['exit_code'] != 0
            or not review_receipt['passed'] or review_receipt['resource_guard_aborted']):
        raise ValueError('Original complete full replay, all full arrays and independently recalculated metrics required')
    base = ROOT/config['base_publication']
    original_bytes = (base/'all_algorithm_dataset_metrics.csv').read_bytes()
    reader = csv.DictReader(io.StringIO(original_bytes.decode('utf-8-sig')))
    before, headers = list(reader), reader.fieldnames
    if len(before)!=8618 or len(headers)!=31 or sum(bool(r['mean']) for r in before)!=2120:
        raise ValueError('Preserve all8618registered metric slots and31fields')
    captured = datetime.now(timezone.utc).isoformat()
    metrics = proof['independently_recomputed_metrics']
    added=[]
    for protocol,key in METRICS:
        values = metrics if protocol=='author_threshold_PA' else metrics['pointwise_no_PA_same_author_threshold']
        row={k:'' for k in headers}
        row.update(algorithm=ALGORITHM,dataset='PSM',task='anomaly_detection',protocol=protocol,metric=key,
            n='1',required_repeats='6',mean=str(values[key]),captured_utc=captured,
            author_equivalence_certified='False',strict_TAB_result='False',source=config['full_replay_proof'],
            comparison='separate_author_protocol_not_strict_leaderboard',epochs='3',metric_unit='fraction',
            selection_boundary='Original historical mirror: test used for early stopping and pooled score calibration; PA only in named protocol;41test tail points dropped.',
            uncertainty_scope='One of6original independent training seeds; no STD,SE,CI or full-cohort robustness claim. Same-seed recovery and readonly replay count once.')
        added.append(row)
    rows=before+added
    check_delta(before,rows)
    target=ROOT/config['output_directory']
    target.mkdir(parents=True,exist_ok=False)
    suffix=io.StringIO(newline='')
    writer=csv.DictWriter(suffix,fieldnames=headers);writer.writerows(added)
    if not original_bytes.endswith(b'\n'):
        raise ValueError('The preserved CSV needs its original complete trailing row boundary')
    with (target/'all_algorithm_dataset_metrics.csv').open('xb') as stream:
        stream.write(original_bytes+suffix.getvalue().encode('utf-8'))
    boundary=('All8,618previous rows and capture times remain unchanged. Append8metrics from the original '
      'Pi historical mirror source-runtime PSM seed42 after complete native model replay and independent '
      'reread of all15full score streams. PA F1=.9803681732; same-threshold pointwise F1=.02237950259, '
      'AUROC=.4812813239. Original test-selection/calibration and41dropped tail points retained. '
      'One of6original seeds; no seed uncertainty or strictTAB/journal-equivalence claim. '
      'All45papers/681method-baseline-ablation/273original-data-task obligations remain.')
    (target/'all_metrics.md').write_text('# Algorithm-dataset metrics v42\n\n'+boundary+'\n\n'
        +legacy.table(headers,[[r.get(k) or '-' for k in headers] for r in rows])+'\n',encoding='utf-8')
    (target/'results.html').write_text(interactive_table(rows,headers),encoding='utf-8')
    (target/'README.rst').write_text('Algorithm-dataset metrics v42\n============================\n\n'+boundary
        +'\n\n8,626metric rows;2,128measured;6,498pending. CSV,Markdown and HTML expose all31fields. '
        'All previously published versions remain immutable.\n',encoding='utf-8')
    exclusive_json(target/'publication_validation.json',dict(published_utc=captured,
        base_publication=config['base_publication'],base_csv_sha256=digest(base/'all_algorithm_dataset_metrics.csv'),
        original_metric_rows_preserved=8618,appended_metric_rows=8,aggregate_metric_rows=len(rows),
        measured_metric_rows=2128,missing_metric_rows=6498,fields=31,
        base_csv_prefix_byte_identical=(target/'all_algorithm_dataset_metrics.csv').read_bytes().startswith(original_bytes),
        previous_rows_and_capture_times_unchanged=True,source_receipts=config['source_receipts'],
        original_scope=config['original_scope'],all_original_experiments_complete=False,
        author_equivalence_certified=False,strict_TAB_result=False,
        outputs={p.name:digest(p) for p in target.iterdir() if p.is_file()}))
    print(json.dumps(dict(rows=len(rows),measured=2128,pending=6498,original_rows_unchanged=8618,appended=8)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    publish(parser.parse_args().config)
