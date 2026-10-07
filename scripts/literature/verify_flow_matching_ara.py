"""Check source provenance, honest result status and all ARA cross-layer bindings."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

from scripts.literature.build_flow_matching_ara import ROOT, inspect_source, read, validate_annotations, write


def verify(root, config, check_sources=True):
    issues=[]; output=root/config['artifact_root']
    index=read(output/'index.json')
    expected={p['id'] for p in config['papers']}
    if expected!={p['id'] for p in index['papers']} or index['paper_count']!=len(expected):
        issues.append('Index coverage differs from registry')
    for paper in config['papers']:
        target=output/'papers'/paper['id']
        try:
            if read(target/'metadata.json')!=paper: raise ValueError('Artifact metadata drift')
            source=read(target/'evidence/source/source_manifest.json')
            if source['anchors']!=paper['anchors'] or source['primary_source_sha256']!=paper['primary_source_sha256']:
                raise ValueError('Source anchor/edition provenance drift')
            if len(source['sources'])!=len(paper['sources']) or any(
                any(actual.get(k)!=v for k,v in expected.items())
                for actual,expected in zip(source['sources'],paper['sources'])):
                raise ValueError('Source manifest differs from configuration')
            if check_sources:
                pages={s['sha256']:inspect_source(root,s) for s in paper['sources']}
                validate_annotations(paper,pages)
            table=read(target/'evidence/tables/reported_results.json')
            if table['results']!=paper['reported_results'] or table['status']!='paper_claims_only':
                raise ValueError('Reported table status/content drift')
            run=read(target/'evidence/runs/local_validation.json')
            if run['E03']['status']!='not_executed_by_this_iteration' or run['E04']['status']!='planned':
                raise ValueError('Engineering artifact has fabricated run status')
            result=subprocess.run([sys.executable,str(root/'scripts/mtsad/validate_ara.py'),'--artifact',str(target)],capture_output=True,text=True)
            if result.returncode: raise ValueError(result.stdout+result.stderr)
        except (ValueError,KeyError,OSError) as error:
            issues.append(paper['id']+': '+str(error))
    return {'paper_count':len(expected),'source_checks':check_sources,'issues':issues,
            'status':'passed' if not issues else 'failed',
            'boundary':'Structure, source identity and honest experiment status checked; not a performance reproduction gate.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/reproducibility/flow_matching_ara.v1.json')
    parser.add_argument('--skip-local-sources',action='store_true',help='Portable structural check; explicitly does not verify local PDF identity.')
    args=parser.parse_args();config=read(args.config)
    report=verify(ROOT,config,not args.skip_local_sources)
    write(ROOT/config['artifact_root']/'validation.json',report)
    print(json.dumps(report,ensure_ascii=True,indent=2));return bool(report['issues'])


if __name__=='__main__': raise SystemExit(main())
