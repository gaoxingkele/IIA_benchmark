"""Replay all original PhysioNet checkpoints on their original CUDA device.

This paired forensic run keeps strict failed CPU evidence and unchanged numeric
tolerances. Only the inference device, execution guards and truthful metadata
change from the complete v2 patient diagnostic replay.
"""
import argparse
import ast
from datetime import datetime, timezone
import os
import subprocess
import time

import psutil

from scripts.flow_matching import audit_saits_physio_patient_discrepancies_v2 as cpu
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write
from scripts.flow_matching.full_gpu_memory_slot_v2 import full_gpu_memory_slot
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.run_grasp_protocol_queue import gpu_snapshot


def worker_ast():
    tree = cpu.replay_ast()
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
    class OriginalDevice(ast.NodeTransformer):
        counts = dict(visibility=0, inference=0, model=0, check=0, metadata=0, boundary=0)
        def visit_Assign(self, node):
            if ast.dump(node) == ast.dump(ast.parse("os.environ['CUDA_VISIBLE_DEVICES'] = '-1'").body[0]):
                self.counts['visibility'] += 1
                return None
            if ast.dump(node) == ast.dump(ast.parse("inference = dict(native, device='cpu')").body[0]):
                self.counts['inference'] += 1
                return ast.parse('inference = dict(native)').body[0]
            return self.generic_visit(node)
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'to' and node.args == []:
                return node
            if (isinstance(node.func, ast.Attribute) and node.func.attr == 'to'
                    and len(node.args) == 1 and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == 'cpu'):
                self.counts['model'] += 1
                node.args[0] = ast.parse("native['device']", mode='eval').body
            if isinstance(node.func, ast.Name) and node.func.id == 'dict':
                keys = {k.arg for k in node.keywords}
                if 'full_checkpoint_contents_and_re_inference_passed' in keys:
                    for keyword in node.keywords:
                        if keyword.arg == 'cuda_context_initialized':
                            assert isinstance(keyword.value, ast.Constant) and keyword.value.value is False
                            keyword.value = ast.Constant(value=True); self.counts['metadata'] += 1
                        elif keyword.arg == 'boundary':
                            keyword.value = ast.Constant(value='All ten original validation-selected checkpoints re-inferred on original CUDA device with full original batches, test grid and unchanged strict tolerance. Every patient discrepancy remains recorded and makes strict reproduction fail. Original CPU forensic evidence is preserved; matching this run does not by itself prove a causal backend explanation, prediction bitwise equivalence, original BRITS2018, author-paper or TAB equivalence.')
                            self.counts['boundary'] += 1
                    node.keywords.append(ast.keyword(arg='replay_inference_device', value=ast.Constant(value='cuda')))
            return node
        def visit_If(self, node):
            if ast.dump(node.test) == ast.dump(ast.parse('torch.cuda.is_initialized()', mode='eval').body):
                if len(node.body) != 1 or not isinstance(node.body[0], ast.Raise):
                    raise ValueError('Keep the exact original CUDA context check')
                self.counts['check'] += 1
                return ast.parse("if not torch.cuda.is_initialized():\n    raise ValueError('Original CUDA checkpoint re-inference must initialize CUDA')").body[0]
            return self.generic_visit(node)
    adapter = OriginalDevice(); worker = adapter.visit(worker)
    if adapter.counts != dict(visibility=1,inference=1,model=1,check=1,metadata=1,boundary=1):
        raise ValueError('Only original-device execution and truthful device metadata may change')
    return ast.fix_missing_locations(ast.Module(body=[worker],type_ignores=[]))


def binding(config):
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']: raise ValueError('Frozen original-device replay source changed')
    prior = read(ROOT / config['cpu_replay_configuration'])
    keys = ['completed_jobs','full_model_count','python','cpu_threads','metric_tolerance']
    if any(config[k] != prior[k] for k in keys) or config['full_model_count'] != 10:
        raise ValueError('Every original full checkpoint, native environment, CPU count and tolerance required')
    if any(config['settings'][k] != value for k,value in prior['settings'].items()):
        raise ValueError('Original physical/commit/RSS/private safeguards must not change')
    if config['resource_lock'] != 'experiments/runs/fm_grasp_protocol_gpu.lock':
        raise ValueError('Serialize original CUDA replay with all native GPU pipelines')
    receipt = read(ROOT / prior['state_root'] / 'receipt.json')
    report = read(ROOT / prior['artifact_directory'] / 'checkpoint_replay_audit.json')
    from scripts.flow_matching.verify_saits_physio_patient_discrepancies_v2 import check_terminal
    check_terminal(report,receipt,prior)
    for item in config['completed_jobs']:
        native = read(ROOT / item['config'])
        if native['device'] != 'cuda' or native.get('diagnostic') or native.get('limit_test_cases'):
            raise ValueError('Original complete CUDA job configuration must remain untouched')
        if sha(ROOT / item['result_path']) != item['result_sha256']:
            raise ValueError('Original complete trained result changed')
    return prior


def worker(config,path):
    binding(config)
    namespace = dict(vars(cpu.legacy.original), patient_statistics_review=cpu.patient_statistics_review)
    exec(compile(worker_ast(),cpu.legacy.original.__file__,'exec'),namespace)
    namespace['worker'](config,path)


def abort_reason(memory,rss,private,gpu,settings):
    if memory['commit_available_bytes'] < settings['emergency_commit_headroom_bytes']: return 'commit'
    if rss > settings['maximum_worker_rss_bytes']: return 'rss'
    if private > settings['maximum_worker_private_bytes']: return 'private'
    if gpu is not None and gpu['gpu_free_bytes'] < settings['emergency_gpu_free_bytes']: return 'gpu'
    return None


def guarded_wait(child,config,api,update):
    settings = config['settings']
    usage = dict(peak_tree_rss_bytes=0,peak_tree_private_bytes=0,resource_guard_aborted=False,
        gpu_observation_retries=0,minimum_gpu_free_bytes=None)
    try:
        while child.poll() is None:
            memory=api['resource_snapshot'](); gpu=None
            try: gpu=gpu_snapshot(settings)
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError):
                usage['gpu_observation_retries'] += 1
            except RuntimeError:
                usage.update(resource_guard_aborted=True,abort_reason='gpu_identity')
                api['terminate_tree'](child); break
            if gpu is not None:
                usage['minimum_gpu_free_bytes'] = gpu['gpu_free_bytes'] if usage['minimum_gpu_free_bytes'] is None else min(gpu['gpu_free_bytes'],usage['minimum_gpu_free_bytes'])
            rss=private=0
            try:
                parent=psutil.Process(child.pid)
                infos=[p.memory_info() for p in [parent,*parent.children(recursive=True)]]
                rss=sum(i.rss for i in infos); private=sum(getattr(i,'private',i.vms) for i in infos)
                usage['peak_tree_rss_bytes']=max(usage['peak_tree_rss_bytes'],rss)
                usage['peak_tree_private_bytes']=max(usage['peak_tree_private_bytes'],private)
            except (psutil.NoSuchProcess,psutil.AccessDenied): pass
            reason=abort_reason(memory,rss,private,gpu,settings)
            if reason is not None:
                usage.update(resource_guard_aborted=True,abort_reason=reason,abort_snapshot=memory)
                api['terminate_tree'](child); break
            update(dict(state='replaying_all_original_cuda_checkpoints',**memory,**usage))
            time.sleep(1)
        return child.wait(),usage
    except BaseException:
        if child.poll() is None: api['terminate_tree'](child)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config',required=True)
    parser.add_argument('--worker',action='store_true'); args=parser.parse_args()
    config=read(ROOT/args.config); binding(config)
    if args.worker: return worker(config,args.config)
    api=resource_api(); api=dict(api,gpu_snapshot=gpu_snapshot)
    base=ROOT/config['state_root']; base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None: raise RuntimeError('Original CUDA replay controller already live')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),state='starting',active_pid=None)
    def update(values): state.update(values,captured_utc=datetime.now(timezone.utc).isoformat()); write(base/'status.json',state)
    try:
        with full_gpu_memory_slot(api,config,update):
            command=[config['python'],'-X','utf8','-u','-m',
                'scripts.flow_matching.audit_saits_physio_original_cuda_replay_v3','--config',args.config,'--worker']
            env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',
                PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS=str(config['cpu_threads']),MKL_NUM_THREADS=str(config['cpu_threads']))
            env.pop('CUDA_VISIBLE_DEVICES',None)
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                handle=psutil.Process(child.pid)
                update(dict(active_pid=child.pid,active_create_time=handle.create_time(),actual_command=handle.cmdline()))
                code,usage=guarded_wait(child,config,api,update)
        path=ROOT/config['artifact_directory']/'checkpoint_replay_audit.json'
        passed=bool(code==0 and not usage['resource_guard_aborted'] and path.exists()
            and read(path)['full_checkpoint_contents_and_re_inference_passed'])
        write(base/'receipt.json',dict(exit_code=code,full_native_model_replay_passed=passed,**usage))
        update(dict(state='completed' if passed else 'failed_preserved',active_pid=None,full_native_model_replay_passed=passed))
    finally: owner.close()


if __name__ == '__main__': main()
