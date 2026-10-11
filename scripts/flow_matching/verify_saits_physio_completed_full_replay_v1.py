"""Independently verify all original patient-coordinate replays, preserving old jobs."""
import ast
from pathlib import Path

from scripts.flow_matching import verify_saits_completed_full_replay_batch_v1 as original
from scripts.flow_matching.validate_saits_legacy_physio_full_result_v1 import legacy_dataset_sha256


def verify_physio_identity(record,native,result,registered):
    if (native['id']!=registered['id'] or record['id']!=registered['id']
            or native['id']!=f"saits_paper_{native['method']}_physio_seed{native['seed']}"
            or native['seed'] not in range(26,31) or native['method'] not in ('saits','brits')
            or native['phase']!='author_original_protocol' or record['config']!=registered['config']
            or result['config']!=native or record['seed']!=native['seed']
            or result['bootstrap_group']!='patient' or native.get('bootstrap_group','patient')!='patient'
            or native.get('diagnostic') or native.get('limit_test_cases') or result['status']!='completed'
            or record['model_parameters']!=result['parameters'] or record['held_out_coordinates']!=result['target_count']
            or record['original_full_dataset_sha256']!=result['dataset_sha256']
            or native.get('dataset_sha256',result['dataset_sha256'])!=result['dataset_sha256']
            or record['validation_best']!=result['validation_best']):
        raise ValueError('Every unchanged original PhysioNet method/patient/seed/checkpoint identity must match')


def review_ast():
    tree=ast.parse(Path(original.__file__).read_text(encoding='utf-8'))
    class PatientBindings(ast.NodeTransformer):
        changed={k:0 for k in ['validator','statistics','groups','dataset_metadata','rate_metadata','dispatch','producer','state_label']}
        def visit_ImportFrom(self,n):
            if n.module=='scripts.flow_matching.validate_saits_full_result_v1':
                n.module='scripts.flow_matching.validate_saits_legacy_physio_full_result_v1';self.changed['validator']+=1
            return n
        def visit_Subscript(self,n):
            if isinstance(n.value,ast.Name) and isinstance(n.slice,ast.Constant):
                if n.value.id=='identities' and n.slice.value=='test_groups':
                    n.slice.value='test_ids';self.changed['groups']+=1
                elif n.value.id=='native' and n.slice.value in ('dataset','missing_ratio'):
                    key=n.slice.value;label='dataset_metadata' if key=='dataset' else 'rate_metadata'
                    self.changed[label]+=1
                    args=[ast.Constant(value=key)]
                    if key=='dataset':args.append(ast.Constant(value='physio'))
                    return ast.copy_location(ast.Call(func=ast.Attribute(value=ast.Name(id='native',ctx=ast.Load()),attr='get',ctx=ast.Load()),args=args,keywords=[]),n)
            return self.generic_visit(n)
        def visit_Constant(self,n):
            substitutions={'group_statistics.npz':('patient_statistics.npz','statistics'),
                'scripts.flow_matching.verify_saits_completed_full_replay_batch_v1':('scripts.flow_matching.verify_saits_physio_completed_full_replay_v1','dispatch'),
                'audit_saits_full_checkpoint_replay_v1':('audit_saits_physio_full_checkpoint_replay_v1','producer'),
                'waiting_actual_full11_native_replay_receipt':('waiting_actual_full10_native_physio_replay_receipt','state_label')}
            if isinstance(n.value,str) and n.value in substitutions:
                value,key=substitutions[n.value];n.value=value;self.changed[key]+=1
            return n
    adapter=PatientBindings();tree=adapter.visit(tree)
    expected=dict(validator=1,statistics=2,groups=1,dataset_metadata=1,rate_metadata=1,dispatch=1,producer=1,state_label=1)
    if adapter.changed!=expected:raise ValueError('Only the registered historical patient bindings and isolated dispatch may change')
    return ast.fix_missing_locations(tree)


def main():
    namespace=dict(__name__='_isolated_saits_physio_complete_array_review',__file__=original.__file__)
    exec(compile(review_ast(),original.__file__,'exec'),namespace)
    namespace['verify_record_identity']=verify_physio_identity
    namespace['main']()


if __name__=='__main__':main()
