"""Apply reviewed, narrowly scoped fixes to separate author working copies."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/runs/flow_matching_campaign/sources"


def sha(payload: bytes):
    return hashlib.sha256(payload).hexdigest()


def patch_text(text: str, before: str, after: str):
    if text.count(before) != 1:
        raise ValueError("Expected exactly one reviewed source pattern; upstream changed")
    return text.replace(before, after, 1)


def corrected_giflow(text: str):
    text = patch_text(text,
        "        for batch in test_loader:\n            batch = batch.to(device)\n            x0, x, y, u, eval_mask, t = preprocess_fm(batch, H_spatial, H_temporal, args.window)\n            preprocessed_val_batches.append((batch, x0, x, y, u, eval_mask, t))\n", "")
    text = patch_text(text, "    ema = EMA(model, decay=0.9999)\n", "    ema = EMA(model, decay=0.9999)\n    ema.register()\n")
    text = patch_text(text, "                    ema.register()\n                    ema.update()", "                    ema.update()")
    text = patch_text(text, "    model.eval()\n    ema.apply_shadow()\n",
        "    model.load_state_dict(torch.load(early_stopping.path, map_location=device, weights_only=True))\n    model.eval()\n")
    loaders = "        train_loader = dm.train_dataloader(batch_size=args.batch_size, shuffle=False)\n        val_loader = dm.val_dataloader(batch_size=args.batch_size, shuffle=False)\n        test_loader = dm.test_dataloader(batch_size=args.batch_size, shuffle=False)\n"
    text = patch_text(text, loaders, "")
    text = patch_text(text, '    if args.model_state == "train":\n',
        ''.join(line[4:] for line in loaders.splitlines(True)) + '    if args.model_state == "train":\n')
    return text


def working_copy(paper_id: str):
    directory = BASE / paper_id
    original, working = directory / "original", directory / "corrected"
    report = directory / "patch_manifest.json"
    if report.exists():
        prior = json.loads(report.read_text(encoding="utf-8"))
        if prior.get('audit_revision') == 4:
            for item in prior['patches']:
                if sha((working / item['file']).read_bytes()) != item['corrected_sha256']:
                    raise ValueError('Working file changed outside the registered patch; preserve it')
            return prior
        for item in prior['patches']:
            if sha((working / item['file']).read_bytes()) != item['corrected_sha256']:
                raise ValueError('Working file changed outside the registered patch; preserve it')
    elif working.exists():
        raise FileExistsError("Unregistered working copy; do not overwrite it")
    else:
        shutil.copytree(original, working)
    records = []
    edits = []
    if paper_id == "cfmi":
        def typing_any(text):
            if text.count('pl.Any') != 2 or 'from typing import Any' not in text:
                raise ValueError('Unexpected CFMI annotation layout')
            return text.replace('pl.Any', 'Any')
        for module in ('cfm', 'csdi'):
            edits.append((f'imp_cfm/models/{module}.py', typing_any,
                          'Fix invalid pl.Any runtime annotations to imported typing.Any; no numerical algorithm change. Confirmed import error under author-declared Lightning 2.2.0.'))
        def restart_arg(text):
            return patch_text(text, "    # Add general arguments\n",
                "    parser.add_argument('--resume_checkpoint', type=str, default=None)\n\n    # Add general arguments\n")
        def restart_fit(text):
            return patch_text(text, '    trainer.fit(model=model, datamodule=datamodule)\n',
                '    trainer.fit(model=model, datamodule=datamodule, ckpt_path=hparams.resume_checkpoint)\n')
        edits.append(('train_argparser.py', restart_arg, 'Add optional Lightning checkpoint restart argument; defaults to the original behavior.'))
        edits.append(('train.py', restart_fit, 'Forward optional checkpoint to Trainer.fit to resume optimizer/scheduler/RNG state after interruption.'))
    if paper_id == "giflow":
        edits.append(("main.py", corrected_giflow,
                      "Remove test batches from validation/early stopping; initialize EMA once; evaluate saved validation-selected checkpoint. Corrected protocol is distinct from author-original scores."))
    if paper_id == "csdi":
        def lazy_attention(text):
            text = patch_text(text, "from linear_attention_transformer import LinearAttentionTransformer\n", "")
            return patch_text(text, "def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):\n",
                "def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):\n  from linear_attention_transformer import LinearAttentionTransformer\n")
        edits.append(("diff_models.py", lazy_attention,
                      "Lazy-load optional linear attention; the paper's is_linear=False Transformer and state dict remain unchanged."))
    if paper_id == "saits":
        def physio_compat(text):
            text = patch_text(text, 'from tsdb import pickle_dump\n',
                'import pickle\n\ndef pickle_dump(obj, path):\n    with open(path + ".pkl", "xb") as handle:\n        pickle.dump(obj, handle, protocol=4)\n')
            text = patch_text(text, 'for filename in os.listdir(args.raw_data_path):',
                'for filename in sorted(os.listdir(args.raw_data_path)):')
            return patch_text(text, '    logger.info(f"All done. Saved to {dataset_saving_dir}.")',
                '    np.savez_compressed(os.path.join(dataset_saving_dir, "audit_arrays.npz"),\n'
                '        train_ids=train_set["RecordID"].to_numpy().reshape(-1, 48)[:, 0],\n'
                '        val_ids=val_set["RecordID"].to_numpy().reshape(-1, 48)[:, 0],\n'
                '        test_ids=test_set["RecordID"].to_numpy().reshape(-1, 48)[:, 0],\n'
                '        feature_names=np.asarray(feature_names), mean=scaler.mean_, scale=scaler.scale_)\n'
                '    logger.info(f"All done. Saved to {dataset_saving_dir}.")')
        edits.append(('dataset_generating_scripts/gene_PhysioNet2012_dataset.py', physio_compat,
            'Replace unused TSDB dependency with metadata-only stdlib pickle export; freeze previously unspecified patient directory ordering lexically; export H5 row IDs/features/scaler for audit. Historical author file ordering is unavailable, so exact split alignment remains unverified. Hourly mean, 37 features, seed 26, train-only scaling and legacy replacement masks are unchanged.'))
        edits.append(('run_models.py', lambda text: patch_text(text, '%Y-%m-%d_T%H:%M:%S', '%Y-%m-%d_T%H-%M-%S'),
            'Use Windows-valid timestamp directory names; no training or selection change.'))
    for relative, function, reason in edits:
        path = working / relative
        before = (original / relative).read_text(encoding="utf-8")
        after = function(before)
        path.write_text(after, encoding="utf-8")
        difference = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile="original/" + relative, tofile="corrected/" + relative))
        patch_file = directory / (relative.replace("/", "_") + ".patch")
        patch_file.write_text(difference, encoding="utf-8")
        records.append({"file": relative, "original_sha256": sha((original / relative).read_bytes()),
                        "corrected_sha256": sha(path.read_bytes()), "reason": reason,
                        "patch_path": str(patch_file.relative_to(ROOT))})
    result = {"paper_id": paper_id, "audit_revision": 4, "status": "reviewed_fixes_applied_not_score_verified",
              "patches": records, "working_path": str(working.relative_to(ROOT))}
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("papers", nargs="+")
    args = parser.parse_args()
    for paper in args.papers:
        print(json.dumps(working_copy(paper)))
