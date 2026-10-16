"""Finish the already-created packet after locating the RST project entrypoint.

The first packaging pass completed all data and validation writes, then stopped
because it looked for README.md instead of the actual project README.rst.
Preserve that packaging source and every produced evidence file; only finish
the entrypoint, integrity inventory and exact Git publication manifest.
"""
import json
from pathlib import Path
import shutil
import subprocess
from scripts.flow_matching import audit_industrial_full466_native_ranges_v2 as s
from scripts.flow_matching import audit_industrial_full466_native_ranges_v1 as failed
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

c=read(ROOT/s.CONFIG);s.verify(c)
name='industrial_all466_full_native_range_metrics_v2'
out=ROOT/'projects/flow_matching_research/results/2026-10-16'/name
assert out.is_dir() and not (out/'packet_integrity.json').exists()
assert read(out/'report.json')==read(Path(c['artifact_root'])/'report.json')
assert read(out/'actual_guard.json')['passed']
assert read(out/'completion_boundary.json')['new_complete_model_results']==0
for path in Path(c['artifact_root']).iterdir():assert sha(path)==sha(out/path.name)
source=ROOT/'tmp/package_industrial466_full_native_ranges_20261016_v2.py'
assert sha(source)==sha(out/'publication_source.py')
project=ROOT/'projects/flow_matching_research/README.rst'
text=project.read_text(encoding='utf-8');assert name not in text
text+='\n`工业全部466项完整TAB区间指标核验 <results/2026-10-16/'+name+'/README.md>`_\n'
text+='6990条固定验证阈值区间指标、16218条实体记录、89个完整五种子组的1335条统计，其中42条未定义保持空白。全部实体/250阈值/完整整数缓冲区及原始指标来源已核验；同分数的点级/区间指标差异单独解释，不据此归因原论文有误。107项相关测试通过，首版过严浮点核验失败记录保留；不新增训练或积分重算完成数。原CPU550项仍缺84项，主总表v68保持原快照，整体任务未完成。\n'
project.write_text(text,encoding='utf-8',newline='\n')
shutil.copyfile(Path(__file__),out/'publication_finalization_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]))
for r in read(out/'packet_integrity.json')['files']:assert sha(out/r['path'])==r['sha256']
ara=ROOT/'projects/flow_matching_research/ara/execution/industrial466_full_native_range_metrics.2026-10-16_v2.json'
assert read(ara)['cases']==466
previous_root=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
previous_parent=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent/'flow-matching-research',text=True).strip()
paths=[p.relative_to(ROOT).as_posix() for p in sorted(out.iterdir()) if p.is_file()]
paths += [s.CONFIG,failed.CONFIG,s.MODULE.replace('.','/')+'.py',failed.MODULE.replace('.','/')+'.py',s.TEST,failed.TEST,
          ara.relative_to(ROOT).as_posix(),project.relative_to(ROOT).as_posix()]
write(ROOT/'tmp/industrial466_full_native_range_publication_manifest_20261016_v2.json',dict(previous_root=previous_root,previous_parent=previous_parent,paths=paths))
print(json.dumps(dict(packet=out.relative_to(ROOT).as_posix(),files=len(paths),cases=466,complete_groups=89,
    metric_rows=6990,new_complete_model_results=0,previous_root=previous_root,previous_parent=previous_parent)))
