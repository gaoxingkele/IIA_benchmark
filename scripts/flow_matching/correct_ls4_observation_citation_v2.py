"""Correct pre-launch literature metadata in a new immutable runtime registration."""
from datetime import datetime,timezone
import json
from scripts.flow_matching.run_ls4_physio_full_observation_dispatch_v1 import ROOT,read,sha,verify,all_jobs_unstarted
from scripts.flow_matching.ls4_physio_full_evaluation_observation_v1 import exclusive


def main():
    previous='configs/runtime/fm_ls4_physio_full_observation_dispatch.v1.json'
    path='configs/runtime/fm_ls4_physio_full_observation_dispatch.v2.json'
    prior=read(ROOT/previous);queue=verify(prior)
    if not all_jobs_unstarted(queue) or (ROOT/prior['state_root']/'status.json').exists():
        raise ValueError('Only pre-launch metadata correction; preserve all frozen running experiments')
    source=queue['source_root']+'/README.md';text=(ROOT/source).read_text(encoding='utf-8')
    title='Deep Latent State Space Models for Time-Series Generation'
    if title not in text or 'https://arxiv.org/abs/2212.12749' not in text:
        raise ValueError('Exact downloaded author README citation required')
    config=dict(prior,runtime_configuration=path,citation=title+'. Zhou, Poli, Xu, Massaroli and Ermon. ICML2023.',
        paper='https://arxiv.org/abs/2212.12749',superseded_prelaunch_metadata_configuration=previous,
        superseded_prelaunch_metadata_configuration_sha256=sha(ROOT/previous))
    config['source_receipts']=prior['source_receipts']+[
        dict(path=previous,sha256=sha(ROOT/previous)),
        dict(path='scripts/flow_matching/correct_ls4_observation_citation_v2.py',sha256=sha(ROOT/'scripts/flow_matching/correct_ls4_observation_citation_v2.py'))]
    verify(config);exclusive(ROOT/path,config)
    exclusive(ROOT/'projects/flow_matching_research/results/2026-10-12/ls4_physio_full_observation_v1/citation_corrected_registration.v2.json',
        dict(captured_utc=datetime.now(timezone.utc).isoformat(),configuration=path,configuration_sha256=sha(ROOT/path),
            source_pins=len(config['source_receipts']),previous_configuration=previous,previous_sha256=sha(ROOT/previous),
            previous_literal_citation=prior['citation'],corrected_citation=config['citation'],paper=config['paper'],
            author_README=source,author_README_sha256=sha(ROOT/source),
            scientific_jobs_arguments_settings_guards_and_observation_code_unchanged=True,
            frozen_previous_configuration_preserved=True,native_training_started_or_stopped=False,
            new_numeric_results_published=0,all_original_experiments_complete=False))
    print(json.dumps(dict(configuration=path,source_pins=len(config['source_receipts']),only_literature_metadata_corrected=True)))


if __name__=='__main__':main()
