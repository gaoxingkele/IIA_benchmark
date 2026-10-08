"""Verify primary-edition anchors and candidate paths without running vendors."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.flow_matching.audit_code_coverage import contained, normal
from scripts.literature.build_flow_matching_ara import ROOT, read, write


def verify(root, inventory, ara):
    for key in ('all_methods_complete','all_ablations_complete','all_experiment_tables_reviewed'):
        if inventory[key]:
            raise ValueError('Source inventory cannot certify exhaustive completion: '+key)
    papers={p['id']:p for p in ara['papers']}
    if {p['paper_id'] for p in inventory['papers']}!=set(papers):
        raise ValueError('Inventory paper coverage differs from ARA')
    pages={}; source_checks=[]
    for ident,paper in papers.items():
        sha=paper['primary_source_sha256']
        source=next(s for s in paper['sources'] if s['sha256']==sha)
        if hashlib.sha256(contained(root,source['path']).read_bytes()).hexdigest()!=sha:
            raise ValueError('Primary PDF SHA256 drift: '+ident)
        cache=read(contained(root,ara['local_text_cache'])/ident/(sha+'.json'))
        if cache['source_sha256']!=sha:
            raise ValueError('Cached edition drift: '+ident)
        pages[ident]=cache['pages']
        source_checks.append({'paper_id':ident,'pdf_sha256_matches':True,'cache_sha256_matches':True})
    identifiers=set(); roles=Counter(); main=Counter()
    for record in inventory['records']:
        ident=record['paper_id']; label=record['record_id']
        if label in identifiers:raise ValueError('Duplicate record: '+label)
        identifiers.add(label)
        if ident not in papers or record['primary_source_sha256']!=papers[ident]['primary_source_sha256']:
            raise ValueError('Record edition mismatch: '+label)
        page=pages[ident].get(str(record['pdf_page']), '')
        if not record['marker'] or normal(record['marker']) not in normal(page):
            raise ValueError('Record page anchor mismatch: '+label)
        if record['role'] not in {'main','baseline','ablation'}:raise ValueError('Unknown record role: '+label)
        roles[record['role']]+=1
        if record['role']=='main':main[ident]+=1
        for value in record['matched_code_paths']+record['model_configs']:
            if not contained(root,value).exists():raise ValueError('Missing candidate path: '+value)
    if any(main[ident]!=1 for ident in papers):raise ValueError('Expected one main artifact per paper')
    if dict(roles)!=inventory['record_counts_by_role']:raise ValueError('Record counts drift')
    return {'source_checks':source_checks,'missing_local_paths':[], 'unique_record_ids':True,
            'evidence_markers_checked_against_declared_primary_page':True,
            'records_are_not_performance_results':True,'paper_count':len(papers),
            'record_count':len(identifiers),'status':'passed'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/reproducibility/fm_paper_method_inventory.v1.json')
    args=parser.parse_args(); inventory=read(args.config)
    result=verify(ROOT,inventory,read(ROOT/inventory['ara_config']))
    write(ROOT/'projects/flow_matching_research/ara/paper_method_validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='source_checks'},indent=2))


if __name__=='__main__':main()
