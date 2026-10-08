import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from scripts.flow_matching import run_crossad_author_job as original
from iia_benchmark.models.fm_crossad_memory_v2 import build_memory_model
from iia_benchmark.models.fm_crossad_author_runtime import import_author


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--queue',type=Path,required=True);parser.add_argument('--job-id',required=True);args=parser.parse_args()
    queue=json.loads(args.queue.read_text());settings=queue['settings'];job=next(j for j in queue['jobs'] if j['id']==args.job_id)
    module=import_author(ROOT/settings['runtime_root'])
    def build(self):return build_memory_model(ROOT/settings['runtime_root'],job['model_parameters']).float()
    module.Exp_Anomaly_Detection._build_model=build
    original.run(job,settings)


if __name__=='__main__':main()
