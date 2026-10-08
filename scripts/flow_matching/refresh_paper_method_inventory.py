"""Attach identity-reviewed source candidates to the curated paper inventory.

Exact declared aliases only; exclusions and component status survive the join.
This does not invent experiment rows or certify method equivalence.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

from scripts.literature.build_flow_matching_ara import ROOT, read, write
from scripts.flow_matching.verify_paper_method_inventory import verify


def matching_candidates(record,source):
    if record['paper_id'] in source.get('excluded_paper_bindings',[]):return []
    method=record['method'].casefold(); matches=[]
    for role,key in [('source_candidate','aliases'),('maintainer_component_candidate','related_component_aliases')]:
        if method in {alias.casefold() for alias in source.get(key,[])}:matches.append(role)
    for scoped in source.get('scoped_aliases',[]):
        if method==scoped['alias'].casefold() and record['paper_id'] in scoped['paper_ids']:
            matches.append('reference_identity_reviewed_source_candidate')
    return matches


def refresh(root=ROOT):
    path=root/'configs/reproducibility/fm_paper_method_inventory.v1.json'
    inventory=read(path); ara_path=root/inventory['ara_config']; ara=read(ara_path)
    inventory.setdefault('ara_config_sha256_at_creation',inventory['ara_config_sha256'])
    inventory['ara_config_sha256']=hashlib.sha256(ara_path.read_bytes()).hexdigest()
    inventory['binding_refresh_date']='2026-10-08'
    coverage=read(root/'configs/reproducibility/fm_code_coverage.v1.json')
    inventory['source_resources']=coverage['source_resources']
    candidates=[]
    for group in ('foundation','tsad','imputation'):
        registry='configs/acquisition/fm_'+group+'_baseline_sources_2026-10-08.json'
        if not (root/registry).exists():continue
        for source in read(root/registry)['sources']:
            if not source.get('original_path'):continue
            candidates.append((source,registry))
    papers={p['id']:p for p in ara['papers']}
    for record in inventory['records']:
        if record['role']=='main':
            record['model_configs']=[c['path'] for c in papers[record['paper_id']]['model_configs']]
        if record['role']!='baseline':continue
        for source,registry in candidates:
            matches=matching_candidates(record,source)
            if not matches:continue
            role=matches[0]
            value=source['original_path']
            if not (root/value).is_dir():raise ValueError('Unacquired alias candidate: '+value)
            if value not in record['matched_code_paths']:record['matched_code_paths'].append(value)
            locators=record.setdefault('candidate_provenance',[])
            locator={'path':value,'role':role,'registry':registry,'repository':source['repository'],
                     'commit':source['commit'],'identity_status':source.get('provenance_status',source.get('source_kind')),
                     'equivalence_status':'not_certified','runtime_status':'not_validated_by_inventory',
                     'boundary':source.get('boundary',source.get('identity_evidence','Candidate source only'))}
            locators[:]=[item for item in locators if item['path']!=value]; locators.append(locator)
            record['coverage']='candidate_source_present_method_equivalence_unverified'
    validation=verify(root,inventory,ara); write(path,inventory)
    report_path=root/'docs/reports/fm_paper_method_inventory_2026-10-08.json'
    report=read(report_path); baselines=[r for r in inventory['records'] if r['role']=='baseline']
    report.update(inventory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),validation=validation,
        baseline_records_with_candidate_code=sum(bool(r['matched_code_paths'] or r['model_configs']) for r in baselines),
        baseline_records_without_verified_candidate_mapping=sum(not(r['matched_code_paths'] or r['model_configs']) for r in baselines),
        unmapped_baseline_labels=dict(Counter(r['method'] for r in baselines if not(r['matched_code_paths'] or r['model_configs']))),
        binding_refresh_date='2026-10-08')
    write(report_path,report)
    output=root/'projects/flow_matching_research/ara'
    write(output/'paper_method_inventory.json',inventory)
    write(output/'paper_method_validation.json',validation)
    for ident in papers:
        records=[record for record in inventory['records'] if record['paper_id']==ident]
        write(output/'papers'/ident/'src/code/method_inventory.json',{
            'paper_id':ident,'primary_source_sha256':papers[ident]['primary_source_sha256'],
            'record_counts_by_role':dict(Counter(record['role'] for record in records)),
            'records':records,'all_methods_complete':False,'all_ablations_complete':False,
            'boundary':'Selected primary-page records and candidate source locators; not exhaustive experiment or runtime acceptance.'})
    return {k:report[k] for k in ('paper_count','record_counts_by_role','baseline_records_with_candidate_code','baseline_records_without_verified_candidate_mapping','complete')}


if __name__=='__main__':print(json.dumps(refresh(),indent=2))
