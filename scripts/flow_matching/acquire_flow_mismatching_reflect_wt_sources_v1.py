"""Complete immutable REFLECT/WT material acquisition, not target adapter identity."""
import argparse
from datetime import datetime,timezone
import os
import shutil
import subprocess

import psutil
from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT,read,sha,verify_receipts
from scripts.flow_matching.acquire_flow_mismatching_optimizer_sources_v1 import verified_extract
from scripts.flow_matching.acquire_flow_mismatching_native_guidance_sources_v1 import verify_pdf
from scripts.flow_matching.acquire_flow_mismatching_visa_v1 import download,json_write


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();cfg=read(ROOT/args.config);verify_receipts(cfg['source_receipts'])
    asset=ROOT/cfg['asset_root']
    if asset.exists():raise FileExistsError('Preserve every previous original code/PDF acquisition attempt')
    if shutil.disk_usage(asset.parent).free<cfg['minimum_free_disk_bytes']:
        raise OSError('Full original source material disk requirement unmet')
    asset.mkdir(parents=True,exist_ok=False)
    state_root=ROOT/cfg['state_root'];state_root.mkdir(parents=True,exist_ok=False)
    process=psutil.Process()
    state=dict(pid=process.pid,create_time=process.create_time(),complete=False,active_pid=None,downloads=[])
    def update(value):
        state.update(value,captured_utc=datetime.now(timezone.utc).isoformat());json_write(state_root/'status.json',state)
    update(dict(phase='starting_full_original_material_acquisition'))
    try:
        downloaded={entry['id']:download(cfg,entry,state,update) for entry in cfg['sources']}
        repositories=[]
        for repo in cfg['repositories']:
            update(dict(phase='auditing_all_original_git_blobs',repository=repo['url']))
            git_root=ROOT/repo['git_root']
            subprocess.run(['git','init','--bare',str(git_root)],check=True)
            subprocess.run(['git','-C',str(git_root),'fetch','--depth','1',repo['url'],repo['commit']],check=True)
            subprocess.run(['git','-C',str(git_root),'fsck','--full'],check=True)
            files=verified_extract(downloaded[repo['source_id']],ROOT/repo['code_root'],git_root,repo['commit'],repo['name'])
            repositories.append(dict(repository=repo['url'],commit=repo['commit'],git_fsck_passed=True,
                original_code_root=str((ROOT/repo['code_root']).resolve()),all_original_git_files_verified=len(files),files=files))
        papers=[verify_pdf(downloaded[name]) for name in cfg['paper_source_ids']]
        report=dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),configuration=args.config,
            configuration_sha256=sha(ROOT/args.config),downloads=state['downloads'],repositories=repositories,original_papers=papers,
            all_original_regular_git_files_and_pdf_pages_verified=True,
            Flow_Mismatching_original_image_domain_adaptation_identity_verified=False,
            full_original_training_executed=False,new_original_training_seeds=0,benchmark_performance=False,
            original_scope=cfg['original_scope'],boundary=cfg['boundary'])
        output=ROOT/cfg['output_directory'];output.mkdir(parents=True,exist_ok=False)
        path=output/'full_original_reflect_wt_material_audit.json';json_write(path,report)
        update(dict(phase='completed_all_original_reflect_wt_material_audits',complete=True,active_pid=None,report=str(path)))
        print(dict(passed=True,repositories=len(repositories),all_original_files=sum(x['all_original_git_files_verified'] for x in repositories),original_papers=len(papers)))
    except Exception as error:
        update(dict(phase='failed_original_material_attempt_preserved',complete=False,error=repr(error)));raise


if __name__=='__main__':main()
