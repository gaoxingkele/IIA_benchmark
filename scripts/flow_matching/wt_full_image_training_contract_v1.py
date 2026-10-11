"""Original2000epoch image-domain WT comparison and explicit native/local gap."""
from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import (
    ROOT,read,sha,validate_job,completion_valid as full_completion_valid)

WT_PARAMETERS=dict(epsilon=1e-5,variance='population',normalization_axes=['C','H','W'],
    transport_direction='normal_to_gaussian',training_time='independent_uniform_unit_time',
    training_endpoint='independent_full_image_standard_gaussian')


def validate_wt_job(job,model,optimizer,dataset):
    n=validate_job(job,model,optimizer,dataset)
    if (job['method']!='WT_Flow_controlled_image_domain' or job['dataset']!='mvtec'
            or job['variant']!='conditional' or dataset['expected_native_splits']!={'train':3629,'test':1725}
            or dataset['expected_original_category_count']!=15 or n!=3629
            or model['method_parameters']!=WT_PARAMETERS
            or job.get('original_WT_best_feature_setting_reproduced') is not False):
        raise ValueError('Full original WT image-domain controlled settings required')
    native=read(ROOT/job['native_components_configuration'])['wt']
    if model['native_source']!={k:native[k] for k in ('source_path','expected_sha256')}:
        raise ValueError('Original immutable WT reverse-flow source identity differs')
    if (sha(ROOT/native['source_path'])!=native['expected_sha256']
            or native['related_native_LayerNorm_epsilon']!=WT_PARAMETERS['epsilon']):
        raise ValueError('Registered related native WT source/statistics changed')
    return n


def completion_valid(job,result,root=ROOT):
    if (result.get('method')!='WT_Flow_controlled_image_domain'
            or result.get('original_WT_best_feature_setting_reproduced') is not False
            or result.get('native_components_configuration_sha256')!=sha(root/job['native_components_configuration'])):
        raise ValueError('Explicit full WT image-domain completion/source boundary required')
    return full_completion_valid(job,result,root)
