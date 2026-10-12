"""Verify the actual complete native RNG gate and retained order artifact."""
import ast
import hashlib
import json
from pathlib import Path
import struct
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'projects/flow_matching_research/results/2026-10-13/rcgan_seed1105_full338_recovery_v1/random_chain'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def array(archive,name,integers=False):
    with archive.open(name+'.npy') as stream:
        prefix=stream.read(8)
        assert prefix[:6]==b'\x93NUMPY' and prefix[6:8] in (b'\x01\x00',b'\x02\x00')
        size=struct.unpack('<H' if prefix[6]==1 else '<I',stream.read(2 if prefix[6]==1 else 4))[0]
        header=ast.literal_eval(stream.read(size).decode('latin1'))
        assert not header['fortran_order']
        if not integers:return header
        assert header['descr']=='<i4' and len(header['shape'])==1
        values=stream.read();assert len(values)==4*header['shape'][0]
        return list(struct.unpack('<'+'i'*header['shape'][0],values))


def test_actual_guarded_native_gate_passed_and_configuration_is_exact():
    proof=read(BASE/'random_order_chain_audit.json')
    config=read(ROOT/proof['configuration'])
    receipt=read(ROOT/config['state_root']/'receipt.json')
    state=read(ROOT/config['state_root']/'status.json')
    assert receipt['exit_code']==0 and receipt['random_order_chain_passed'] and not receipt['resource_guard_aborted']
    assert state['state']=='completed_native_gate' and state['native_gate_passed'] and state['active_pid'] is None
    assert proof['passed'] and proof['configuration_sha256']==sha(ROOT/proof['configuration'])
    assert proof['verified_original_numpy_version']=='1.18.5' and proof['job']['seed']==1105
    assert config['completed_original_epochs']==proof['original_completed_epochs']==338


def test_all338_original_epoch_random_states_and4984_point_evaluator_bindings_verified():
    proof=read(BASE/'random_order_chain_audit.json')
    records=proof['verified_epoch_records']
    assert [r['epoch'] for r in records]==list(range(338))
    assert proof['complete_full_rng_and_cumulative_data_order_verified']
    assert proof['original_full_evaluator_points_per_epoch']==4984
    assert all(r['full4984_noise_and_real_pixels_bitwise_verified'] and r['post_shuffle_numpy_and_python_states_bitwise_verified'] for r in records)
    assert proof['protocol_draw_counts']==dict(D=338*1284,G=338*1284,loss=338)
    assert all(len(r[k])==64 for r in records for k in ['evaluation_sha256','numpy_state_sha256','python_state_sha256'])


def test_full_order_archive_hash_shapes_and_cumulative_permutation_are_consistent():
    proof=read(BASE/'random_order_chain_audit.json');artifact=proof['derived_order_artifact']
    assert sha(artifact['path'])==artifact['sha256']
    with zipfile.ZipFile(artifact['path']) as archive:
        expected={'cumulative_train_order':(36000,), 'cumulative_train_image_ids':(36000,),
            'initial_split_permutation':(60000,), 'initial_visualization_noise':(28,28,5),
            'initial_visualization_image_ids':(6,), 'original_epoch_loss_noise':(338,28,28,5),
            'original_epoch_loss_batch_image_ids':(338,28), 'original_full_evaluator_image_ids':(338,4984)}
        assert set(archive.namelist())=={name+'.npy' for name in expected}
        assert all(array(archive,name)['shape']==shape for name,shape in expected.items())
        order=array(archive,'cumulative_train_order',True)
        identities=array(archive,'cumulative_train_image_ids',True)
        initial=array(archive,'initial_split_permutation',True)
    assert sorted(order)==list(range(36000)) and sorted(initial)==list(range(60000))
    assert identities==[initial[i] for i in order]


def test_rng_proof_does_not_certify_model_optimizer_restore_training_or_full500_result():
    proof=read(BASE/'random_order_chain_audit.json')
    assert proof['required_original_epochs']==500 and 500-proof['original_completed_epochs']==162
    assert proof['tensorflow_model_updates_executed_by_this_audit']==0 and not proof['numerical_training_losses_reconstructed']
    assert not proof['full500_epoch_result_verified'] and not proof['benchmark_performance']
    assert not proof['author_equivalence_certified'] and not proof['all_original_experiments_complete']
    assert proof['additional_training_seed_slots']==0
    assert proof['original_scope']==dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273)
