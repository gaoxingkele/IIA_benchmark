"""Independently reread every original CUDA replay, retaining strict failures."""
import ast

from scripts.flow_matching import verify_saits_physio_patient_discrepancies_v2 as cpu
from scripts.flow_matching.audit_saits_physio_original_cuda_replay_v3 import binding
from scripts.flow_matching.audit_saits_full_checkpoint_replay_v1 import ROOT,read,sha


def check_terminal(replay,receipt,producer):
    if replay.get('cuda_context_initialized') is not True or replay.get('replay_inference_device') != 'cuda':
        raise ValueError('Require actual full original CUDA model re-inference')
    # All original complete-model/coordinate/strict-failure checks remain intact.
    cpu.check_terminal(dict(replay,cuda_context_initialized=False),receipt,producer)


def verify_registration(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path']) != pin['sha256']: raise ValueError('Frozen full CUDA review source changed')
    producer=read(ROOT/config['replay_configuration']); prior=binding(producer)
    if (config['settings'] != prior['settings'] or config['resource_lock'] != prior['resource_lock']
            or config['python'] != producer['python'] or config['required_full_models'] != 10):
        raise ValueError('Keep original full10 native CPU array-review resources and model count')
    return producer


def review_ast():
    tree=cpu.review_ast()
    class DeviceBindings(ast.NodeTransformer):
        dispatches=0
        producers=0
        def visit_Constant(self,node):
            if node.value == 'scripts.flow_matching.verify_saits_physio_patient_discrepancies_v2':
                node.value='scripts.flow_matching.verify_saits_physio_original_cuda_replay_v3'; self.dispatches+=1
            elif node.value == 'audit_saits_physio_patient_discrepancies_v2':
                node.value='audit_saits_physio_original_cuda_replay_v3'; self.producers+=1
            return node
    adapter=DeviceBindings(); tree=adapter.visit(tree)
    if (adapter.dispatches,adapter.producers) != (1,1):
        raise ValueError('Only original-device isolated dispatch and producer identity may change')
    return ast.fix_missing_locations(tree)


def main():
    namespace=dict(__name__='_full_original_cuda_patient_array_review',
        __file__=cpu.legacy.original.__file__,patient_statistics_review=cpu.patient_statistics_review)
    exec(compile(review_ast(),cpu.legacy.original.__file__,'exec'),namespace)
    namespace['verify_record_identity']=cpu.legacy.verify_physio_identity
    namespace['check_terminal']=check_terminal
    namespace['verify_registration']=verify_registration
    namespace['main']()


if __name__ == '__main__': main()
