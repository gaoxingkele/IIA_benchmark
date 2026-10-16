"""Resolve the active original table and paper-text/author-log recipe conflict."""
import ast
from datetime import datetime,timezone
import json
from pathlib import Path
import re
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive

AUDIT='projects/flow_matching_research/ara/adopted_baselines/trajectorynet_paper_text_vs_logged_recipe.2026-10-16_v1.json'
SOURCE='configs/acquisition/fm_TrajectoryNet_arxiv_source.v1.json'
MATERIAL='configs/acquisition/fm_TrajectoryNet_author_checkpoint_materials.v1.json'


def active_text(text):
    return '\n'.join(re.sub(r'(?<!\\)%.*','',line) for line in text.splitlines())


def table_rows(text):
    result={}
    for line in text.splitlines():
        if '&' not in line:continue
        label=line.split('&')[0].strip().replace(' ','')
        if label not in ['Base','Base+D','Base+V','Base+D+V','Base+E','Base+G','OT','prev','next','rand']:continue
        numbers=re.findall(r'(?<![A-Za-z])\d+\.\d+',line.split('&',1)[1])
        if len(numbers)!=4:raise ValueError('Every active table row must contain four scalar targets')
        result[label]=list(map(float,numbers))
    return result


def generate():
    config,material=read(ROOT/SOURCE),read(ROOT/MATERIAL)
    receipt=read(Path(config['output_directory'])/'verified_acquisition.json')
    root=Path(receipt['text_directory'])
    if not receipt['passed'] or sha(Path(receipt['archive']))!=receipt['archive_sha256']:
        raise ValueError('Complete original versioned source archive required')
    for row in receipt['members']:
        if row['extracted'] and sha(root/row['path'])!=row['sha256']:
            raise ValueError('Original source member hash changed')
    chain=['icml.tex','sections/main.tex','sections/6_experiments.tex','tables/eb_table.tex','tables/eb_table_v5.tex']
    for current,following in zip(chain,chain[1:]):
        inputs=re.findall(r'\\(?:input|include)\s*\{([^}]+)\}',active_text((root/current).read_text(encoding='utf-8')))
        if following not in inputs:raise ValueError('Only the actually included Table3 may define targets')
    targets=table_rows((root/chain[-1]).read_text(encoding='utf-8'))
    previous=read(ROOT/'projects/flow_matching_research/ara/adopted_baselines/trajectorynet_original_recipe_and_metric_audit.2026-10-16_v1.json')
    if targets!=previous['paper']['table3']:raise ValueError('Active TeX table must match all40 published scalar targets')
    body=(root/'sections/6_experiments.tex').read_text(encoding='utf-8').splitlines()[1]
    if 'LeakyReLU' not in body or r'weight decay $5 \times 10^{-5}$' not in body:
        raise ValueError('Exact active experiment recipe declaration required')
    from pypdf import PdfReader
    paper_receipt=read(Path(material['output_directory'])/'verified_acquisition.json')
    pdf=Path(paper_receipt['transfers']['paper']['path'])
    page=PdfReader(pdf).pages[6].extract_text()
    if 'LeakyReLU' not in page or 'weight decay 5' not in page:
        raise ValueError('Source assertion must also be visible in the actual original PDF')
    profile=read(ROOT/'configs/models/fm_TrajectoryNet_author_fig8_EB_leave3_replay.v1.json')
    if profile['source_parameters']['nonlinearity']!='tanh' or profile['source_parameters']['weight_decay']!=1e-5:
        raise ValueError('Actual released author log recipe changed')
    author_root=ROOT/material['source_root']
    activation=author_root/'TrajectoryNet/lib/layers/odefunc.py'
    assignment=next(n for n in ast.parse(activation.read_text()).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='NONLINEARITIES' for t in n.targets))
    original_options=[ast.literal_eval(k) for k in assignment.value.keys]
    if any('leaky' in k.lower() for k in original_options):raise ValueError('Unexpected author LeakyReLU implementation')
    appendix=(root/'sections/appendices.tex').read_text(encoding='utf-8').splitlines()
    grid_lines=[i+1 for i,line in enumerate(appendix) if r'\lambda_{energy}' in line and 'search' in line]
    artificial_lines=[i+1 for i,line in enumerate(appendix) if r'\lambda_j = 1' in line]
    references=[dict(path=str(root/n),sha256=sha(root/n)) for n in chain+['sections/appendices.tex']]
    audit=dict(schema_version=1,checked_at_utc=datetime.now(timezone.utc).isoformat(),
        evidence_kind='original_source_include_graph_and_full_target_recipe_audit_not_model_performance',
        citation='https://arxiv.org/abs/2002.04461v2',official_source_url=receipt['source_url'],
        original_source_archive_sha256=receipt['archive_sha256'],original_source_archive_bytes=receipt['archive_bytes'],
        original_source_members=51,verified_extracted_text_members=sum(r['extracted'] for r in receipt['members']),
        publisher_checksum_available=False,source_execution_performed=False,
        active_Table3_include_chain=chain,active_Table3_all40_scalars_match_pdf_audit=True,active_Table3=targets,
        unused_old_table_files_not_promoted=['tables/kantorovich_table_v3.tex','tables/eb_table_v4.tex','tables/eb_table_v6.tex'],
        published_recipe=dict(pdf_page=7,tex_file='sections/6_experiments.tex',tex_line=2,
            activation='LeakyReLU',weight_decay=5e-5,maximum_iterations=10000,batch_size=1000,
            solver='dopri5',atol=1e-5,rtol=1e-5,learning_rate=.001,
            negative_slope_specified=False,hidden_widths_specified=False),
        released_author_log_recipe=dict(activation='Tanh',weight_decay=1e-5,hidden_widths=[64,64,64],
            source_checkpoint='results/fig8_results/checkpt.pth',historical_training_seed_unknown=True),
        original_activation_options=original_options,original_source_implements_paper_LeakyReLU=False,
        added_paper_text_control=dict(configuration='configs/models/fm_TrajectoryNet_EB_paper_text_full_grid.v1.json',
            local_negative_slope_assumption=.01,inherited_author_hidden_widths=[64,64,64],
            exact_historical_recipe_identity_certified=False,author_log_recipe_preserved=True,
            activation_and_weight_decay_effects_are_not_individually_identifiable=True),
        energy_recipe=dict(grid_tex_lines=grid_lines,artificial_example_Jacobian_lines=artificial_lines,
            EB_specific_Jacobian_weight_or_selected_grid_cell_found=False,
            no_artificial_example_coefficient_transplant=True,all45_energy_conditions_per_protocol_remain_pending=True),
        source_bindings=references+[dict(path=activation.relative_to(ROOT).as_posix(),sha256=sha(activation)),
            dict(path=str(pdf),sha256=sha(pdf)),dict(path=SOURCE,sha256=sha(ROOT/SOURCE))],
        original_scope=previous['original_scope'],all_original_experiments_complete=False)
    exclusive(ROOT/AUDIT,audit)
    metadata=dict(material['paper'],arxiv_id='2002.04461',arxiv_version='v2',publication_type='conference',
        full_text_status='open_pdf',pdf_url=material['paper']['url'],local_pdf_path=str(pdf),
        download_status='downloaded',download_source='arxiv',source_platforms=['arxiv'],fetched_at='2026-10-16',
        supplementary_source_path=receipt['archive'],supplementary_source_sha256=receipt['archive_sha256'])
    exclusive(ROOT/config['state_root']/'paper_metadata.json',metadata)
    print(json.dumps(dict(audit=AUDIT,active_Table3_scalars=40,source_members=51,
        paper_activation='LeakyReLU',paper_weight_decay=5e-5,author_activation='Tanh',author_weight_decay=1e-5,
        energy_Jacobian_coefficient_still_unknown=True)))


if __name__=='__main__':generate()
