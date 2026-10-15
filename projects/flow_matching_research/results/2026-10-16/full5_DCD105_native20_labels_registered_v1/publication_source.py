import json,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import xml.etree.ElementTree as ET
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.flow_matching import replay_full5_DCD105_min105_controls_v1 as s
folder=ROOT/'projects/flow_matching_research/results/2026-10-16/full5_DCD105_native20_labels_registered_v1'
assert {p.name for p in folder.iterdir()}=={'README.md'}

def copy(source,name):
    target=folder/name;assert not target.exists();shutil.copyfile(source,target);assert s.sha(source)==s.sha(target)

c=s.read(ROOT/s.CONFIG);s.verify(c)
validation=Path('F:/aicoding/IIA_Data/experiments/full5_DCD105_min105_native_and20_labels_tests_20261016_v1.xml')
xs=list(ET.parse(validation).getroot().iter('testsuite'))
assert sum(int(x.get('tests','0')) for x in xs)==24 and all(int(x.get(k,'0'))==0 for x in xs for k in ['failures','errors','skipped'])
copy(validation,'validation24.xml')
paths=[s.MODULE.replace('.','/')+'.py',s.CONFIG,s.TEST]
for index,path in enumerate(paths):copy(ROOT/path,f'original{index:02d}'+Path(path).suffix)
proofs=[]
fixtures=Path('F:/aicoding/IIA_Data/experiments/full5_DCD105_min105_native_test_fixtures_v1')
for path in fixtures.glob('test_full_local3epoch_256wi*/fixture_metadata.json'):
    m=s.read(path);base=path.parent;identity=m['case']['id'];row={}
    assert m['fixture_only_not_benchmark'] and m['model']['parameters']['epochs']==3
    for phase in s.REPORTS:
        file=(base/'replay'/('scalar/'+identity if phase=='scalar' else identity+'/'+phase)/s.REPORTS[phase])
        p=s.read(file);assert p['passed'];row[phase]=dict(path=str(file),sha256=s.sha(file))
        if phase=='native':
            assert p['additional_fit_calls']==p['additional_optimizer_updates']==0
            assert p['all_array_dtype_shape_content_bytes_identical'] and p['all_model_tensors_identical_before_and_after']
            assert p['all_source_forward_and_FP64_formula_observation']['windows']==12
            assert sum(x['points'] for x in p['full_native_coverage'])==858
        copy(file,'fixture_'+phase+'_proof.json')
    proofs.append(dict(full_original_local_model=m['model'],full_local_epochs=3,proofs=row,
        source_result=dict(path=str(base/'source/result.json'),sha256=s.sha(base/'source/result.json')),
        fixture_only=True,real_benchmark_performance_certified=False))
assert len(proofs)==1
s.write(folder/'actual_full_local3epoch_tiny_CPU_chain_not_benchmark.json',dict(records=proofs))
launch=s.read(ROOT/c['state_root']/'launch.json');state=s.read(ROOT/c['state_root']/'status.json')
p=psutil.Process(launch['pid']);assert p.create_time()==launch['create_time']==state['create_time'] and p.cmdline()==launch['command']
assert not state['failed_replays_preserved'] and not state['source_failures_preserved'] and not state['completed_cases']
s.write(folder/'actual_live_owner_and_original_resource_policies.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    actual_owner=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()),status=state,
    original_native_policy=dict(settings=c['settings'],resource_lock=c['resource_lock']),
    original_scalar_policy=dict(settings=c['scalar_settings'],resource_lock=c['scalar_resource_lock']),
    actual_real_dataset_replays_complete=0,all_original_experiments_complete=False))
copy(__file__,'publication_source.py')
s.write(folder/'archive_validation.json',dict(tests_passed=24,registered_native_cases=5,registered_label_views=20,
    actual_real_dataset_replays_complete=0,full_original_scope=c['original_scope'],all_original_experiments_complete=False,
    files_sha256={p.relative_to(folder).as_posix():s.sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}))
ara=ROOT/'projects/flow_matching_research/ara/execution/full5_DCD105_min105_native20_labels.2026-10-16_v1.json'
s.write(ara,dict(evidence=folder.relative_to(ROOT).as_posix(),registered_native_cases=5,registered_label_views=20,tests_passed=24,
    actual_real_dataset_native_replays_complete=0,original_scope=c['original_scope'],all_original_experiments_complete=False))
paths+=[ara.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()]
git=lambda root,*a:subprocess.check_output(['git','-C',str(root),*a],text=True).strip()
assert not git(ROOT,'diff','--cached','--name-only')
s.write(ROOT/'tmp/full5_DCD105_native20_publication_manifest_20261016_v1.json',dict(
    previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths),tests_passed=24,actual_real_dataset_replays_complete=0,live_pid=p.pid)))
