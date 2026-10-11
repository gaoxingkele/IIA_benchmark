import json

from scripts.flow_matching.repair_saits_publication_display_v51 import render


def test_actual_counts_and_source_version_identities_survive_rendering():
    rows=[{'mean':'0.8','source':'results/v42/original.json'},
        {'mean':'','source':'results/v49/pending.json'},
        {'mean':'0.9','source':'results/v50/verified.json'}]
    page=render(rows,['mean','source'],51,'one seed < five required')
    assert '<p>3 项；2 项有实测值，1 项待完成。</p>' in page
    assert '<title>实验技术指标 v51</title>' in page
    assert 'one seed &lt; five required' in page
    assert json.loads(page.split('<script>const rows=',1)[1].split(', cols=',1)[0])==rows
