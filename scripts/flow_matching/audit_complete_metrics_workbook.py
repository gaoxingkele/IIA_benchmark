import hashlib
import argparse
import json
import math
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--source', default='projects/flow_matching_research/results/2026-10-09/complete_metrics')
parser.add_argument('--manifest', default='outputs/fm_complete_metrics_20261009/workbook_manifest.json')
parser.add_argument('--visually-reviewed', action='store_true')
args = parser.parse_args()
target = root / args.source
data = json.loads((target / 'complete_results.json').read_text(encoding='utf-8'))
manifest = json.loads((root / args.manifest).read_text(encoding='utf-8'))
ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
checked = 0
with zipfile.ZipFile(target / 'complete_experiment_metrics.xlsx') as archive:
    strings = []
    if 'xl/sharedStrings.xml' in archive.namelist():
        strings = [''.join(item.itertext()) for item in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
    def value(cell):
        if cell is None:
            return None
        if cell.get('t') == 'inlineStr':
            return ''.join(cell.itertext())
        child = cell.find('m:v', ns)
        if child is None:
            return None
        if cell.get('t') == 's':
            return strings[int(child.text)]
        if cell.get('t') == 'str':
            return child.text
        if cell.get('t') == 'b':
            return child.text == '1'
        return float(child.text)
    matrices = []
    for i, sheet in enumerate(manifest['sheets'], 1):
        xml = ET.fromstring(archive.read(f'xl/worksheets/sheet{i}.xml'))
        cells = {c.get('r'): value(c) for c in xml.findall('.//m:sheetData/m:row/m:c', ns)}
        rows = xml.findall('.//m:sheetData/m:row', ns)
        assert max(int(r.get('r')) for r in rows) == sheet['rows'] + 5
        assert xml.find('m:sheetViews/m:sheetView/m:pane', ns) is not None
        assert xml.find('m:tableParts', ns) is not None
        matrices.append(cells)
    def equal(i, address, expected):
        global checked
        actual = matrices[i].get(address)
        if expected is None:
            assert actual is None, (i, address, actual, expected)
        elif isinstance(expected, (int, float)):
            assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (i, address, actual, expected)
        else:
            assert actual == expected, (i, address, actual, expected)
        checked += 1
    for i, row in enumerate(data['strict_summary'], 6):
        for column, key in [('A', 'algorithm_config'), ('B', 'dataset'), ('C', 'completed_seeds'),
                            ('D', 'required_seeds'), ('E', 'precision_mean'), ('F', 'recall_mean'),
                            ('G', 'f1_mean'), ('H', 'auroc_mean'), ('I', 'average_precision_mean'),
                            ('J', 'VUS_ROC_mean'), ('K', 'VUS_PR_mean'), ('L', 'affiliation_f_mean'),
                            ('M', 'VUS_ROC_n'), ('N', 'affiliation_f_n'), ('O', 'status')]:
            equal(0, f'{column}{i}', row[key])
    for i, row in enumerate(data['strict_runs'], 6):
        for column, key in [('A', 'algorithm_config'), ('B', 'dataset'), ('C', 'seed'), ('D', 'calibration_false_alarm_percent'),
                            ('E', 'precision'), ('F', 'recall'), ('G', 'f1'), ('H', 'auroc'), ('I', 'average_precision'),
                            ('K', 'VUS_ROC'), ('M', 'affiliation_f'), ('AB', 'parameter_count'), ('AD', 'training_seconds')]:
            equal(1, f'{column}{i}', row.get(key))
    for i, row in enumerate(data['tab_ratio_diagnostic'], 6):
        for column, key in [('E', 'f_score'), ('F', 'adjust_f_score'), ('G', 'auc_roc'), ('H', 'auc_pr')]:
            equal(2, f'{column}{i}', row[key])
    for i, row in enumerate(data['historical_runs'], 6):
        for column, key in [('A', 'model'), ('B', 'dataset'), ('C', 'seed'), ('D', 'protocol'),
                            ('G', 'pointwise_f1'), ('J', 'adjusted_f1')]:
            equal(3, f'{column}{i}', row.get(key))
    formal = sorted(data['imputation']['formal_results'], key=lambda r: (-(r['local_mean'] is not None), r['algorithm'], r['dataset']))
    for i, row in enumerate(formal, 6):
        equal(4, f'G{i}', row['local_mean'])
        equal(4, f'N{i}', row['paper_mean'])
        equal(4, f'R{i}', row['verdict'])
    for i, row in enumerate(data['imputation']['per_run_metrics'], 6):
        equal(5, f'I{i}', row['value'])
    tables = [p for p in archive.namelist() if p.startswith('xl/tables/table') and p.endswith('.xml')]
    assert len(tables) == 9
    assert all(ET.fromstring(archive.read(p)).find('m:autoFilter', ns) is not None for p in tables)
validation = json.loads((target / 'validation.json').read_text(encoding='utf-8'))
validation['workbook'] = {'sheets_verified': len(matrices), 'rows': sum(s['rows'] for s in manifest['sheets']),
                          'exported_cells_compared_to_source': checked, 'all_filter_tables_and_panes_verified': True,
                          'all_sheets_visually_reviewed': args.visually_reviewed}
validation['outputs'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.iterdir()
                         if p.is_file() and p.name != 'validation.json'}
(target / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n', encoding='utf-8')
print(json.dumps(validation['workbook'], indent=2))
