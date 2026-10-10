"""Native TimeGAN budgets/coverage must not silently shrink or imply reproduction."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.flow_matching import timegan_native_common_v1 as common
from scripts.flow_matching.run_timegan_native_job_v1 import flatten_fetches, validate_evaluators


class NativeTimeGANTests(unittest.TestCase):
    def setUp(self):
        self.model = dict(parameters=dict(seq_len=24, hidden_dim=24, num_layer=3, iterations=50000,
                                         batch_size=128, metric_iteration=10))
        self.counts = dict(E0=50000, GS=50000, G=100000, E=100000, D=34921)

    def test_three_complete_stages_accept_conditional_discriminator(self):
        common.verify_counts(self.counts, self.model)
        common.verify_counts(dict(self.counts, D=0), self.model)
        common.verify_counts(dict(self.counts, D=50000), self.model)

    def test_reduced_training_hidden_batch_or_evaluation_rejected(self):
        for key in self.model['parameters']:
            model = copy.deepcopy(self.model)
            model['parameters'][key] -= 1
            with self.assertRaises(ValueError):
                common.full_budget(model)

    def test_missing_joint_or_stage_update_rejected(self):
        for name in ('E0', 'GS', 'G', 'E'):
            with self.assertRaises(ValueError):
                common.verify_counts(dict(self.counts, **{name: self.counts[name]-1}), self.model)
        with self.assertRaises(ValueError):
            common.verify_counts(dict(self.counts, D=50001), self.model)

    def test_native_nested_fetch_observation(self):
        a, b, c = object(), object(), object()
        self.assertEqual(flatten_fetches({'train': [a, (b,)], 'loss': c}), [a, b, c])

    def evaluator_records(self):
        return [dict(kind=k, repeat=r, optimizer_updates=n, score=.2)
                for k, n in [('discriminative', 2000), ('predictive', 5000)] for r in range(10)]

    def test_all_twenty_full_native_evaluators_required(self):
        records = self.evaluator_records()
        validate_evaluators(records)
        with self.assertRaises(ValueError):
            validate_evaluators(records[:-1])
        records[0]['optimizer_updates'] = 1999
        with self.assertRaises(ValueError):
            validate_evaluators(records)

    def test_evaluator_duplicate_and_nonfinite_rejected(self):
        records = self.evaluator_records()
        records[1]['repeat'] = 0
        with self.assertRaises(ValueError):
            validate_evaluators(records)
        records = self.evaluator_records()
        records[0]['score'] = float('nan')
        with self.assertRaises(ValueError):
            validate_evaluators(records)

    def test_complete_dataset_module_seed_coverage_cannot_shrink(self):
        jobs = [dict(id='%s_%s_%s' % (d,m,s), dataset=d, module=m, seed=s)
                for d in ('stock','energy','sine') for m in ('gru','lstm','lstmLN') for s in range(1103,1108)]
        queue = dict(source_receipts=[], model_config='unused', jobs=jobs)
        with patch.object(common,'read',return_value=self.model), patch.object(common,'verify_source'):
            common.verify_queue(queue)
            for changed in [jobs[:-1], jobs[:-1]+[jobs[0]]]:
                with self.assertRaises(ValueError):
                    common.verify_queue(dict(queue, jobs=changed))

    def test_every_untouched_source_file_is_hash_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source'
            source.mkdir()
            files = []
            for i in range(13):
                p = source/str(i)
                p.write_text('source-'+str(i))
                files.append(dict(path=str(i),sha256=common.sha(p)))
            (root/'manifest.json').write_text(json.dumps(dict(commit='frozen',files=files)))
            model = dict(source_root='source',source_manifest='manifest.json',source_commit='frozen')
            with patch.object(common,'ROOT',root):
                common.verify_source(model)
                (source/'12').write_text('modified')
                with self.assertRaises(ValueError):
                    common.verify_source(model)


if __name__ == '__main__':
    unittest.main()
