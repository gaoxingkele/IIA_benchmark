"""Recovery fidelity gates, with the real pinned author training AST."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.flow_matching import timegan_stage_resume_v2 as recovery


class RecoveryFidelityTests(unittest.TestCase):
    def setUp(self):
        self.config = recovery.read(recovery.ROOT / 'configs/reproducibility/fm_timegan_stock_stage_recovery.2026-10-11_v2.json')

    def test_full_author_joint_and_all_other_statements_preserved(self):
        queue, model, job, records = recovery.binding(self.config)
        module = SimpleNamespace(__file__=str(recovery.ROOT / model['source_root'] / 'timegan.py'))
        source = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
        original = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'timegan')
        resumed = recovery.resumed_ast(module).body[0]
        original_loops = [n for n in original.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'itt']
        resumed_loops = [n for n in resumed.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'itt']
        self.assertEqual(len(original_loops), 3)
        self.assertEqual(len(resumed_loops), 1)
        self.assertEqual(ast.dump(original_loops[2]), ast.dump(resumed_loops[0]))
        restored = copy.deepcopy(resumed)
        restored.name = original.name
        restored.body = [n for n in restored.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                         and isinstance(n.value.func, ast.Name) and n.value.func.id == '_verified_restore')]
        positions = [i for i, n in enumerate(original.body) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'itt']
        for index in positions[:2]:
            self.assertIsInstance(restored.body[index], ast.Pass)
            restored.body[index] = copy.deepcopy(original.body[index])
        self.assertEqual(ast.dump(original), ast.dump(restored))
        self.assertEqual(len(queue['jobs']), 45)
        self.assertEqual(records[-1]['optimizer_updates'], dict(E0=50000, GS=50000, G=0, E=0, D=0))

    def test_previous_attempt_cannot_be_overwritten(self):
        config = dict(self.config, output_directory=self.config['canonical_job']['output_directory'])
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            recovery.binding(config)

    def test_checkpoint_integrity_required(self):
        config = copy.deepcopy(self.config)
        config['stage_receipt_sha256']['supervised_complete'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'receipt changed'):
            recovery.binding(config)

    def test_same_canonical_seed_required(self):
        config = copy.deepcopy(self.config)
        config['canonical_job']['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'same canonical'):
            recovery.binding(config)

    def test_tf_stochastic_ancestors_reject_training_rng(self):
        rng = SimpleNamespace(name='noise', type='RandomUniform', inputs=[], control_inputs=[])
        root = SimpleNamespace(name='train', type='NoOp', inputs=[], control_inputs=[rng])
        self.assertEqual(recovery.stochastic_ancestors([root]), [dict(name='noise', type='RandomUniform')])

    def test_frozen_full_native_restore_proof_required(self):
        config = dict(self.config, native_restore_audit_sha256='0' * 64)
        with patch.object(recovery, 'sha', side_effect=lambda p: '1' * 64 if str(p).endswith('full_native_restore_and_transition_audit.json') else original_sha(p)):
            with self.assertRaisesRegex(ValueError, 'restore proof changed'):
                recovery.compile_observed_resume(config)


original_sha = recovery.sha


if __name__ == '__main__':
    unittest.main()
