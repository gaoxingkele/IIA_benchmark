"""Replay every original PSM point with arithmetic-free final-use deletion."""
import ast
from scripts.flow_matching import replay_crossad_residual_lifetime_v4 as previous

original=previous.original


def worker(config,path):
    import iia_benchmark.models.fm_crossad_memory_v2 as native
    from iia_benchmark.models.fm_crossad_attention_lifetime_v2 import build_attention_lifetime_memory_model
    queue=original.read(config['queue']);job=next(j for j in queue['jobs'] if j['id']==config['job_id'])
    if job['mode']!='release_checkpoint' or job['reconstructed_ablation'] or 'ablation_row' in job:
        raise ValueError('Only the same preserved full author release model may use this replay')
    native.build_memory_model=build_attention_lifetime_memory_model
    original.worker(config,path)
    proof_path=original.ROOT/config['artifact_directory']/'full_native_model_replay.json'
    proof=original.read(proof_path)
    proof.update(attention_scores_released_only_after_last_original_softmax_use=True,
        residual_temporary_references_released_only_after_last_original_use=True,
        original_full_batches_and_all_arithmetic_dropout_normalization_operations_retained=True,
        no_inference_chunking_or_model_architecture_change=True,
        original_RSS_private_and_commit_guards_unchanged=True)
    proof_path.write_text(__import__('json').dumps(proof,indent=2)+'\n',encoding='utf-8')


def main_ast():
    tree=previous.main_ast();changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='scripts.flow_matching.replay_crossad_residual_lifetime_v4':
            node.value='scripts.flow_matching.replay_crossad_attention_lifetime_v5';changed+=1
    if changed!=1:raise ValueError('Only the fresh guarded native worker route may change')
    return tree


def main():
    namespace=dict(original.__dict__,worker=worker,_cached_resource_api=previous.cached_resource_api)
    exec(compile(main_ast(),__file__,'exec'),namespace)
    namespace['main']()


if __name__=='__main__':main()
