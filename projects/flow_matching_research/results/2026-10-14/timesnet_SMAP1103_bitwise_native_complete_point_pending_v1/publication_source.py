"""Publish completed native replay without closing the pending point audit."""
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[1]
base='projects/flow_matching_research/results/2026-10-14/timesnet_SMAP1103_bitwise_native_complete_point_pending_v1'
folder=root/base
proof=json.loads((folder/'full_bitwise_native_replay_with_standalone_point_audit_pending.json').read_text())
assert proof['full_original_local_training_and_full_native_score_replay_passed']
assert proof['original_and_replay_score_archives_same_sha256']
assert proof['separately_registered_point_audit_pending'] and not proof['full_registered_two_stage_audit_complete']
assert proof['full_registered_training_windows']==1044 and proof['configured_optimizer_steps_all_epochs_from_original_code']==27
assert proof['validation_tests_passed']==37 and not proof['all_original_experiments_complete']
(folder/'publication_source.py').write_bytes(Path(__file__).read_bytes())
pending='projects/flow_matching_research/results/2026-10-14/timesnet_SMAP1103_constant_fit_full_replay_v1/README.md'
with (root/pending).open('a',encoding='utf-8',newline='\n') as stream:
    stream.write('\nNative-stage completion update: all train/validation/test timestamps now match '
        'bitwise after full checkpoint inference, including the complete compressed score '
        'archive digest. See ../timesnet_SMAP1103_bitwise_native_complete_point_pending_v1/README.md. '
        'The independently registered standalone point stage still awaits original admission; '
        'the earlier pending snapshot is preserved.\n')
targets=[base,pending]
paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    raw=subprocess.check_output(['git',*args,'--',*targets],cwd=root)
    paths.update(p for p in raw.decode().splitlines() if not p.endswith('.log'))
assert paths and not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root).strip()
manifest=dict(previous_root='cb10234bf2632f12d042a6cb464a6aca72fc22e7',
    previous_parent='e18adaa2279afb2663ef919ce7baebaec98e84c9',paths=sorted(paths))
(root/'tmp/TimesNet_SMAP1103_native_complete_point_pending_publication_manifest_20261014_v1.json').write_text(
    json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(exact_files=len(paths),full_bitwise_native_replay_passed=True,
    standalone_point_audit_pending=True,validation_tests_passed=37)))
