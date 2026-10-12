"""Same full400 CFM producer, without eager NumPy/SciPy imports in its parent."""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from scripts.flow_matching.run_light_controller import ROOT,load_functions


def lightweight_namespace():
    values=dict(globals())
    load_functions(ROOT,'scripts/flow_matching/prepare_cfm_ts_original_data.py',['sha','read'],values)
    load_functions(ROOT,'scripts/flow_matching/run_cfm_ts_original_job.py',['verify','fingerprint','write'],values)
    values['original']=SimpleNamespace(ROOT=ROOT,**{k:values[k] for k in ('sha','read','verify','fingerprint','write')})
    source=ast.parse((ROOT/'scripts/flow_matching/cfm_complete_epoch_checkpoint_v1.py').read_text(encoding='utf-8'))
    assignment=next(n for n in source.body if isinstance(n,ast.Assign) and any(getattr(t,'id',None)=='EXECUTION_FIELDS' for t in n.targets))
    values['EXECUTION_FIELDS']=ast.literal_eval(assignment.value)
    load_functions(ROOT,'scripts/flow_matching/cfm_complete_epoch_checkpoint_v1.py',['canonical_science'],values)
    load_functions(ROOT,'scripts/flow_matching/run_cfm_checkpointed_original_job_v1.py',['verify_registration'],values)
    load_functions(ROOT,'scripts/flow_matching/run_cfm_checkpointed_full_retry_v1.py',['main'],values)
    return values


def allowed_idle(role,state,output_exists,descendants):
    from scripts.flow_matching.run_timegan_corrected_observer_dispatch_v2 import allowed_console_child
    return (state.get('active_pid') is None and state.get('retained_partial_epochs_credited')==0
        and state.get('state') in ('waiting_for_memory','waiting_for_heavy_lock')
        and not output_exists
        and all(allowed_console_child(p['name'],p['exe']) for p in descendants))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--handoff-config',required=True)
    args=parser.parse_args();values=lightweight_namespace();config=values['read'](ROOT/args.handoff_config)
    for pin in config['source_receipts']:
        if values['sha'](ROOT/pin['path'])!=pin['sha256']:raise ValueError('Lightweight parent source binding changed')
    producer=values['read'](ROOT/config['producer_configuration']);values['verify_registration'](producer)
    from types import FunctionType
    from scripts.flow_matching import run_timegan_corrected_observer_dispatch_v2 as common
    namespace=dict(common.__dict__,handoff_phase_allowed=allowed_idle)
    handoff=FunctionType(common.safe_handoff.__code__,namespace)
    base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    def update(data):values['write'](base/'status.json',data)
    route=dict(previous_owner=config['previous_owner'],previous_state_path=config['previous_state_path'],
        runtime_configuration=args.handoff_config,output_directory=config['scientific_output_directory'])
    handoff('original',route,base,update,10)
    sys.argv=['same_original_full400_producer','--config',config['producer_configuration']]
    values['main']()


if __name__=='__main__':main()
