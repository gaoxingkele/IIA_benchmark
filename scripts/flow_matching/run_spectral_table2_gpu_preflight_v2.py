"""Retry the full original GPU preflight with the author's relative-data cwd fixed."""
import argparse
import json
import os
from pathlib import Path

from scripts.flow_matching import run_spectral_table2_original_v1 as original
from scripts.flow_matching.freeze_spectral_table2_environment_v1 import ROOT,sha
from scripts.flow_matching.spectral_table2_bootstrap_v1 import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();queue=original.read(ROOT/args.queue)
    original.verify(queue);original.verify_environment(queue)
    output=ROOT/args.output
    if output.exists():raise FileExistsError('Completed preflight proof must remain unchanged')
    original_workspace=original.workspace
    def private_workspace(q,destination):
        # Preserve the failed gpu_workspace; all author source/data bytes remain exact.
        private=original_workspace(q,Path(destination).with_name(Path(destination).name+'_v2'))
        os.chdir(private)
        return private
    original.workspace=private_workspace
    original.preflight_gpu(queue,output)
    proof=original.read(output)
    proof.update(preflight_adapter=Path(__file__).relative_to(ROOT).as_posix(),preflight_adapter_sha256=sha(Path(__file__)),
        windows_cwd_fix='The full eight-worker author loaders run in their private author workspace',
        previous_failed_workspace_preserved=True,scientific_model_batch_budget_or_sampler_changed=False)
    write(output,proof)


if __name__=='__main__':main()
