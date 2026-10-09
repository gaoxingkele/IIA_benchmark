"""Prepare immutable original raw splits and register all full GRASP jobs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from scripts.flow_matching.grasp_protocol_runner import ROOT, environment_versions, sha, verify, write_json


def score_recipes(profile, config):
    default = {'source_samples':5,'flow_evaluations':10}
    if profile != 'main':
        return [default]
    choices = [default]
    choices += [{'source_samples':5,'flow_evaluations':n} for n in config['inference_flow_times']]
    choices += [{'source_samples':n,'flow_evaluations':10} for n in config['inference_source_samples']]
    return [dict(p) for p in {tuple(p.items()):p for p in choices}.values()]


def prepare(config, root=ROOT):
    import numpy as np
    from scripts.flow_matching.prepare_tsad_execution import load_entities
    sys.path.insert(0, str(root / 'src'))
    from iia_benchmark.models.grasp_protocol_v2 import fit_only_scaler
    if config['train_fraction'] != .8 or config['purge_points'] != 0:
        raise ValueError('Chronological 80/20 with no purge required')
    target = root / config['prepared_root']
    if target.exists():
        raise FileExistsError('Prepared data is immutable; choose a new registered version')
    registered = json.loads((root / config['registered_raw_manifest']).read_text(encoding='utf-8'))
    source_sha = {s['path']:s['sha256'] for d in registered['datasets'] for s in d['source_files']}
    data_config = json.loads((root / config['source_data_config']).read_text(encoding='utf-8'))
    specs = {s['id']:s for s in data_config['datasets']}
    target.mkdir(parents=True)
    datasets = []
    for dataset in config['datasets']:
        spec = specs[dataset['id']]
        entities = []
        for name, train, test, labels, paths, audit in load_entities(root, spec):
            raw_sources = []
            for source_path in paths:
                path = Path(source_path)
                relative = path.relative_to(root).as_posix()
                actual = sha(path)
                if actual != source_sha[relative]:
                    raise ValueError('Registered raw file changed: '+relative)
                raw_sources.append({'path':relative,'sha256':actual})
            cut = int(len(train)*config['train_fraction'])
            if min(cut, len(train)-cut, len(test)) < config['base_parameters']['window']:
                raise ValueError('Original full window unsupported by entity: '+name)
            if train.shape[1] != dataset['actual_features'] or test.shape[1] != train.shape[1]:
                raise ValueError('Full feature contract differs')
            if labels.shape != (len(test),) or not np.isin(labels,[0,1]).all():
                raise ValueError('Full original labels differ')
            if audit.get('training_anomaly_points') not in (0, None):
                raise ValueError('Known training anomaly points contradict normal training')
            _, _, scaler = fit_only_scaler(train[:cut], train[cut:])
            destination = target / dataset['id'] / (name+'.npz')
            destination.parent.mkdir(exist_ok=True)
            np.savez_compressed(destination, train=train[:cut], validation=train[cut:], test=test,
                                labels=labels.astype(np.uint8), train_row_ids=np.arange(cut),
                                validation_row_ids=np.arange(cut,len(train)), test_row_ids=np.arange(len(test)))
            for receipt in raw_sources:
                if sha(root / receipt['path']) != receipt['sha256']:
                    raise ValueError('Raw file changed during preparation')
            record = {'id':name,'dataset':dataset['id'],'prepared_path':destination.relative_to(root).as_posix(),
                      'prepared_sha256':sha(destination),'raw_sources':raw_sources,
                      'raw_train_points':len(train),'split_cut':cut,'fit_points':cut,
                      'validation_points':len(train)-cut,'test_points':len(test),
                      'test_anomaly_points':int(labels.sum()),'features':train.shape[1],
                      'paper_variables':dataset['paper_variables'],'feature_count_discrepancy':True,
                      'normal_training_status':'verified_zero_labels' if audit.get('training_anomaly_points') == 0 else 'train_labels_not_provided',
                      'fit_window_count':cut-49,'validation_window_count':len(train)-cut-49,'test_window_count':len(test)-49,
                      'source_audit':audit,'train_only_scaler':{k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in scaler.items()},
                      'boundary_windows_crossed':False,'raw_files_overwritten':False}
            entities.append(record)
            print(json.dumps({'prepared':name,'dataset':dataset['id'],'fit_points':cut,'validation_points':len(train)-cut}), flush=True)
        if len(entities) != dataset['entities']:
            raise ValueError('Original dataset entity count differs')
        datasets.append({**dataset,'entities':entities,'known_train_only_entities':spec.get('known_train_only_entities',[])})
    manifest = {'schema_version':2,'datasets':datasets,'train_fraction':.8,'purge_points':0,
                'preprocessing':'raw chronological split only; fitting-only repair and scale applied by each model',
                'raw_overwrite':False,'source_data_config':config['source_data_config'],
                'source_data_config_sha256':sha(root / config['source_data_config']),
                'registered_raw_manifest_sha256':sha(root / config['registered_raw_manifest'])}
    write_json(root / config['manifest'], manifest)
    return manifest


def register(config, registration_path, root=ROOT):
    output = root / config['queue']
    if output.exists():
        raise FileExistsError('Frozen queue exists')
    manifest = json.loads((root / config['manifest']).read_text(encoding='utf-8'))
    profiles, jobs = {}, []
    for profile in config['profiles']:
        path = f"configs/models/fm_grasp_protocol_v2_{profile['id']}.json"
        if (root / path).exists():
            raise FileExistsError('Model configuration already frozen')
        parameters = {**config['base_parameters'], **profile['overrides']}
        model = {'schema_version':2,'id':'fm_grasp_protocol_v2_'+profile['id'],
                 'family':'flow_matching','task':'anomaly_detection',
                 'entrypoint':'iia_benchmark.models.grasp_protocol_v2:GRASPProtocolDetector',
                 'parameters':parameters,'profile':profile['id'],'citation':config['citation'],
                 'paper_pdf':config['paper_pdf'],'primary_protocol_corrections':config['primary_protocol_corrections'],
                 'reproduction_status':'paper_derived_callable_not_author_equivalent',
                 'tests':['tests/test_grasp_protocol_v2.py','tests/test_grasp_protocol_runner.py'],
                 'author_repository':None,'paper_score_status':'not_verified','boundary':config['boundary']}
        write_json(root / path, model)
        profiles[profile['id']] = path
    # First complete the single-entity SWAN cohorts, then larger full datasets.
    for data in manifest['datasets']:
        for seed in config['seeds']:
            for profile, model_path in profiles.items():
                for entity in data['entities']:
                    ident = f"grasp_protocol_v2__{profile}__{data['id']}__{entity['id']}__seed{seed}"
                    jobs.append({'id':ident,'profile':profile,'dataset':data['id'],'entity':entity['id'],'seed':seed,
                                 'model_config':model_path,'model_config_sha256':sha(root / model_path),
                                 'expected_optimizer_updates':1500*math.ceil(entity['fit_window_count']/256),
                                 'test_points':entity['test_points'],'validation_points':entity['validation_points'],
                                 'score_recipes':score_recipes(profile, config),
                                 'output_directory':config['state_root']+'/jobs/'+ident})
    environment = root / config['environment_lock']
    if environment.exists():
        raise FileExistsError('Environment already frozen')
    write_json(environment, environment_versions())
    paths = [registration_path,config['manifest'],config['source_data_config'],config['paper_pdf'],
             config['primary_protocol_corrections'],config['native_metrics'],config['environment_lock'],
             'src/iia_benchmark/models/grasp_protocol_v2.py','src/iia_benchmark/models/fm_native.py',
             'scripts/flow_matching/grasp_protocol_runner.py','scripts/flow_matching/register_grasp_protocol.py',
             'scripts/flow_matching/run_grasp_protocol_queue.py','scripts/flow_matching/summarize_grasp_protocol.py',
             'scripts/flow_matching/prepare_tsad_execution.py',*profiles.values()]
    paths += ['tests/test_grasp_protocol_v2.py','tests/test_grasp_protocol_runner.py']
    queue = {k:config[k] for k in ('state_root','artifact_root','resource_lock','runtime_config','python','settings',
                                  'environment_lock','manifest','native_metrics','boundary','seeds')}
    queue.update(schema_version=2, registration_config=registration_path, profiles=profiles,jobs=jobs,
                 cohort_count=len(profiles)*len(config['datasets'])*len(config['seeds']),
                 source_receipts=[{'path':p,'sha256':sha(root/p)} for p in sorted(set(paths))])
    verify(queue, root)
    write_json(output, queue)
    print(json.dumps({'full_entity_jobs':len(jobs),'cohorts':queue['cohort_count'],
                      'inference_score_sets':sum(len(j['score_recipes']) for j in jobs),
                      'source_receipts':len(queue['source_receipts'])}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/fm_grasp_protocol_registration.v2.json')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--register-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only and args.register_only:
        parser.error('Select at most one stage')
    config = json.loads((ROOT / args.config).read_text(encoding='utf-8'))
    if not args.register_only:
        prepare(config)
    if not args.prepare_only:
        register(config, args.config)


if __name__ == '__main__':
    main()
