from pathlib import Path
import json
import shutil
import xml.etree.ElementTree as ET
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

packet=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_row5_seed1104_full17_v1'
xml=Path('F:/aicoding/IIA_Data/experiments/CrossAD_full17_component_controls_20261016_v1_tests.xml')
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==15
assert all(int(x.get('failures',0))==int(x.get('errors',0))==int(x.get('skipped',0))==0 for x in suites)
newxml=packet/'relevant_fresh15_full17_validation.xml';assert not newxml.exists();shutil.copyfile(xml,newxml)
path=packet/'README.md';content=path.read_text(encoding='utf-8')
old='附原15项检查XML，验证器未修改，本次完整数组重算不算新的单元测试。'
assert content.count(old)==1
content=content.replace(old,'原检查XML保留；本轮在第17完整条件及30组点值对比核验后，又重新执行相关15项检查全部通过，并随附新的实际XML。原验证器未修改，检查次数不算模型实验次数。')
path.write_text(content,encoding='utf-8')
shutil.copyfile(__file__,packet/'final_validation_source.py')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file() and p.name!='packet_integrity.json']))
manifest=ROOT/'tmp/CrossAD17_component_controls_manifest_20261016_v1.json';m=read(manifest)
m['paths']=sorted(set(m['paths'])|{newxml.relative_to(ROOT).as_posix(),(packet/'final_validation_source.py').relative_to(ROOT).as_posix()})
write(manifest,m)
print(json.dumps(dict(fresh_tests=15,full_CUDA_conditions=17,paired_point_controls=30,files=len(m['paths']))))
