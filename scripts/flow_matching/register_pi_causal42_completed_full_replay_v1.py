"""Freeze all three unchanged full replay stages for the completed PSM42 model."""
import ast
import inspect
from scripts.flow_matching import register_pi_source1105_completed_full_replay_v1 as previous
from scripts.flow_matching import run_pi_causal42_completed_full_replay_v1 as replay


def registration_ast():
    tree=ast.parse(inspect.getsource(previous.main))
    recovery_changes=0;path_changes=0
    for node in ast.walk(tree):
        if not isinstance(node,ast.Constant) or not isinstance(node.value,str):continue
        if node.value=='configs/experiments/fm_resource_recovery_queue.v4.json':
            node.value='configs/experiments/fm_pi_causal42_full_incident_recovery_queue.v5.json';recovery_changes+=1
        if 'pi_source1105_completed_full_replay' in node.value:
            node.value=node.value.replace('pi_source1105_completed_full_replay','pi_causal42_completed_full_replay').replace('2026-10-12/','2026-10-14/')
            path_changes+=1
        if node.value.startswith('One newly finished original source-runtime PSM seed1105,'):
            node.value=node.value.replace('source-runtime PSM seed1105','causal-axis PSM seed42')
    if recovery_changes!=1 or path_changes!=10:
        raise ValueError('Only the selected recovery queue and ten separate metadata publication paths may change: '+str((recovery_changes,path_changes)))
    return ast.fix_missing_locations(tree)


def main():
    values=dict(previous.__dict__,CANONICAL=replay.CANONICAL,RECOVERY=replay.RECOVERY,verify=replay.verify)
    exec(compile(registration_ast(),__file__,'exec'),values)
    values['main']()


if __name__=='__main__':main()
