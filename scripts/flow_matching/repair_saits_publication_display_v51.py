"""Preserve all audited numeric records while correcting counts and embedded source paths."""
import argparse
import csv
from datetime import datetime, timezone
import html
import json
import os
import re
import subprocess

import psutil

from scripts.flow_matching.publish_saits_electricity20_seed26_v50 import ROOT, read, digest, exclusive
from scripts.flow_matching.publish_pi_original_source_listing_v42 import interactive_table


def render(rows, fields, version, boundary):
    page = interactive_table(rows, fields)
    # Restrict version changes to headings. Source paths inside the JSON data
    # must retain their original v42/v49/v50 identities.
    page, title_count = re.subn(r'(<title>[^<]*?)v42(</title>)', rf'\g<1>v{version}\2', page)
    page, heading_count = re.subn(r'(<h1>[^<]*?)v42(</h1>)', rf'\g<1>v{version}\2', page)
    measured = sum(bool(row.get('mean')) for row in rows)
    counts = f'{len(rows):,} 项；{measured:,} 项有实测值，{len(rows)-measured:,} 项待完成。'
    paragraphs = list(re.finditer(r'<p>.*?</p>', page, re.S))
    if title_count != 1 or heading_count != 1 or len(paragraphs) != 2:
        raise ValueError('Exact inherited display headings and two paragraphs required')
    for match, value in reversed(list(zip(paragraphs, (counts, boundary)))):
        page = page[:match.start()] + '<p>' + html.escape(value) + '</p>' + page[match.end():]
    encoded = page.split('<script>const rows=',1)[1].split(', cols=',1)[0]
    if json.loads(encoded) != rows:
        raise ValueError('Every embedded metric/source record must exactly match the CSV')
    return page


def verify(config):
    for pin in config['source_receipts']:
        if digest(ROOT/pin['path']) != pin['sha256']:
            raise ValueError('Frozen display correction input changed')
    base = ROOT/config['base_publication']
    numeric = read(base/'publication_validation.json')
    review = read(base/'independent_publication_review.json')
    if (not review['passed'] or review['publication_receipt_sha256'] != digest(base/'publication_validation.json')
            or numeric['measured_metric_rows'] != 2131 or numeric['aggregate_metric_rows'] != 8626
            or numeric['full_original_seed_cohort_complete'] or numeric['original_full_seed_count'] != 1):
        raise ValueError('Require successful original full-array numeric review and unchanged partial cohort')
    return base


def worker(config):
    base = verify(config)
    with (base/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);rows=list(reader);fields=reader.fieldnames
    if len(rows) != 8626 or len(fields) != 31 or sum(bool(r['mean']) for r in rows) != 2131:
        raise ValueError('Full audited CSV scope required')
    target=ROOT/config['output_directory'];target.mkdir(parents=True,exist_ok=False)
    (target/'all_algorithm_dataset_metrics.csv').write_bytes((base/'all_algorithm_dataset_metrics.csv').read_bytes())
    boundary=('SAITS Electricity 20% 缺失率已完成并完整重推理核验 seed26，当前为1/5个原始训练种子。'
        'MAE 0.81348、RMSE 1.25164、MRE 0.43528。跨种子标准差、标准误和置信区间尚不可估计；'
        '10个月份的分块 bootstrap 仅描述单个训练种子的时间敏感性。论文条件对齐与严格TAB等价性未认证，完整实验目标仍待完成。')
    page=render(rows,fields,51,boundary)
    (target/'results.html').write_text(page,encoding='utf-8')
    text=(base/'all_metrics.md').read_text(encoding='utf-8')
    if not text.startswith('# Algorithm-dataset metrics v50\n'):
        raise ValueError('Exact previous audited full table required')
    (target/'all_metrics.md').write_text(text.replace('# Algorithm-dataset metrics v50\n',
        '# Algorithm-dataset metrics v51\n',1),encoding='utf-8')
    (target/'README.rst').write_text('Algorithm-dataset metrics v51\n============================\n\n'+boundary+
        '\n\nAll8626CSVrecords and31fields are byte-identical to fully independently auditedv50. '
        'Measured2131;pending6495. Display counts are corrected; version replacements affect headings only. '
        'All embedded source paths retain their actual original identities. The historicalv50 display and its '
        'numeric review remain preserved. No new metrics/seeds/model inference or paper/TAB equivalence claim.\n',encoding='utf-8')
    receipt=dict(published_utc=datetime.now(timezone.utc).isoformat(),base_publication=config['base_publication'],
        base_numeric_receipt_sha256=digest(base/'publication_validation.json'),
        base_numeric_independent_review_sha256=digest(base/'independent_publication_review.json'),
        aggregate_metric_rows=8626,fields=31,measured_metric_rows=2131,missing_metric_rows=6495,
        all_numeric_tables_byte_identical_to_base=True,changed_numeric_metric_rows=0,
        original_full_seed_count=1,required_original_seeds=5,full_original_seed_cohort_complete=False,
        original_scope=config['original_scope'],author_equivalence_certified=False,strict_TAB_result=False,
        additional_training_seed_slots=0,all_original_experiments_complete=False,
        source_receipts=config['source_receipts'],outputs={p.name:digest(p) for p in target.iterdir() if p.is_file()})
    exclusive(target/'publication_validation.json',receipt)
    # Verify actual written files and decoded full JSON, not just generated strings.
    stored=(target/'results.html').read_text(encoding='utf-8')
    actual=json.loads(stored.split('<script>const rows=',1)[1].split(', cols=',1)[0])
    if actual != rows or (target/'all_algorithm_dataset_metrics.csv').read_bytes() != (base/'all_algorithm_dataset_metrics.csv').read_bytes():
        raise ValueError('Actual complete numeric and embedded table identities differ')
    if '<p>8,626 项；2,131 项有实测值，6,495 项待完成。</p>' not in stored:
        raise ValueError('Display count must reflect actual complete metric records')
    exclusive(target/'independent_publication_review.json',dict(passed=True,
        captured_utc=datetime.now(timezone.utc).isoformat(),actual_numeric_csv_byte_identity_verified=True,
        all8626embedded_records_and_original_source_paths_equal_to_actual_csv=True,
        actual_display_counts_verified=True,original_full_array_numeric_review=config['base_publication']+'/independent_publication_review.json',
        aggregate_metric_rows=8626,fields=31,measured_metric_rows=2131,missing_metric_rows=6495,
        publication_receipt_sha256=digest(target/'publication_validation.json'),
        full_original_seed_cohort_complete=False,all_original_experiments_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    config=read(ROOT/args.config);verify(config)
    if args.worker:return worker(config)
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    from scripts.flow_matching.run_resource_gated_light_controller_v1 import write
    api=resource_api();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Display correction already live')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None)
    def update(v):state.update(v);write(base/'status.json',state)
    try:
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            cmd=[config['python'],'-X','utf8','-u','-m','scripts.flow_matching.repair_saits_publication_display_v51',
                '--config',args.config,'--worker']
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',
                        PYTHONDONTWRITEBYTECODE='1'),creationflags=subprocess.CREATE_NO_WINDOW)
                p=psutil.Process(child.pid);update(dict(state='correcting_and_verifying_full_display',
                    active_pid=p.pid,active_create_time=p.create_time(),actual_command=p.cmdline()))
                code,usage=capped_wait(child,config,api,update)
        exclusive(base/'receipt.json',dict(exit_code=code,**usage))
        update(dict(state='completed' if code==0 and not usage['resource_guard_aborted'] else 'failed_preserved',active_pid=None))
    finally:owner.close()


if __name__ == '__main__':main()
