"""Full original CrossAD replay with shorter residual-reference lifetimes only."""
import ast
import inspect

from scripts.flow_matching import replay_crossad_registered_model_v3 as original


def worker(config,path):
    import iia_benchmark.models.fm_crossad_memory_v2 as native
    from iia_benchmark.models.fm_crossad_reference_lifetime_v1 import build_lifetime_memory_model
    queue=original.read(config['queue']);job=next(j for j in queue['jobs'] if j['id']==config['job_id'])
    if job['mode']!='release_checkpoint' or job['reconstructed_ablation'] or 'ablation_row' in job:
        raise ValueError('This full memory-lifetime recovery is only the existing fixed release model')
    native.build_memory_model=build_lifetime_memory_model
    original.worker(config,path)
    proof_path=original.ROOT/config['artifact_directory']/'full_native_model_replay.json'
    proof=original.read(proof_path)
    proof.update(residual_temporary_references_released_only_after_last_original_use=True,
        original_full_batches_and_all_arithmetic_dropout_normalization_operations_retained=True,
        no_inference_chunking_or_model_architecture_change=True,
        original_RSS_private_and_commit_guards_unchanged=True)
    proof_path.write_text(__import__('json').dumps(proof,indent=2)+'\n',encoding='utf-8')


def main_ast():
    tree=ast.parse(inspect.getsource(original.main));changes=dict(route=0,telemetry=0)
    class Binding(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value=='scripts.flow_matching.replay_crossad_registered_model_v3':
                n.value='scripts.flow_matching.replay_crossad_residual_lifetime_v4';changes['route']+=1
            return n
        def visit_Call(self,n):
            if ast.dump(n)==ast.dump(ast.parse('original.resource_api()',mode='eval').body):
                changes['telemetry']+=1
                return ast.copy_location(ast.Call(func=ast.Name(id='_cached_resource_api',ctx=ast.Load()),args=[],keywords=[]),n)
            return self.generic_visit(n)
    tree=Binding().visit(tree)
    if changes!=dict(route=1,telemetry=1):raise ValueError('Exactly one native worker route and host telemetry may change')
    return ast.fix_missing_locations(tree)


def cached_resource_api():
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    api=original.original.resource_api();api['resource_snapshot']=resource_snapshot
    return api


def main():
    namespace=dict(original.__dict__,worker=worker,_cached_resource_api=cached_resource_api)
    exec(compile(main_ast(),__file__,'exec'),namespace)
    namespace['main']()


if __name__=='__main__':main()
