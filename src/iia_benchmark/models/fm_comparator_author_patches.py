"""Reproducible repairs to preserved comparator source snapshots.

Only write a fresh corrected directory. Never alter the original snapshot.
Pi journal pp. 5--6 requires a common causal support for series/prior.
These repairs do not certify every journal loss/ablation as reproduced.
"""
from pathlib import Path
import hashlib
import json
import shutil


def prepare_pi_transformer(source: str | Path, destination: str | Path) -> dict:
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or source == destination or source in destination.parents:
        raise ValueError("fresh destination outside source required")
    file = source / "model/attn.py"
    original = file.read_bytes()
    content = original.decode("utf-8")
    edits = [
        ('device="cuda",', 'device="cpu",', "CPU-safe constructor default; forward uses tensor device"),
        ('scores.masked_fill(attn_mask.mask, -np.inf)',
         'scores = scores.masked_fill(attn_mask.mask, -np.inf)',
         "masked_fill was out-of-place and its result was discarded"),
        ('x_flat = q.permute(0, 2, 1, 3).reshape(B, L, -1).detach()',
         'x_flat = q.reshape(B, L, -1).detach()',
         "preserve timestamp axis for Hilbert phase summary"),
        ('prior = prior * gate\n',
         'prior = prior * gate\n        if self.mask_flag:\n            prior = prior.masked_fill(attn_mask.mask, 0.0)\n',
         "journal requires prior and series on identical causal support"),
    ]
    for old, new, reason in edits:
        if content.count(old) != 1:
            raise ValueError(f"source version does not match patch: {reason}")
        content = content.replace(old, new)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    corrected = content.encode("utf-8")
    (destination / "model/attn.py").write_bytes(corrected)
    manifest = {
        "source": str(source), "corrected": str(destination), "file": "model/attn.py",
        "original_sha256": hashlib.sha256(original).hexdigest(),
        "corrected_sha256": hashlib.sha256(corrected).hexdigest(),
        "changes": [e[2] for e in edits],
        "remaining_gaps": ["layer count is e_layers - 1", "Encoder exposes only last-layer regularisers",
                           "phase gate mixes head channels rather than a per-head scalar projection",
                           "full journal ablation recipes are not supplied"],
    }
    (destination / "local_patch_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def prepare_maelnet_reward(source: str | Path, destination: str | Path,
                           false_negative: float = -1.5, false_positive: float = -0.6,
                           include_slow_learner: bool = True) -> dict:
    """MaelNet Tables III/IV: author-code slow/checkpoint and reward variants."""
    if false_negative not in {-1.5, -2.0} or false_positive not in {-0.6, -0.8}:
        raise ValueError("only paper Table IV penalty values accepted")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or source == destination or source in destination.parents:
        raise ValueError("fresh destination outside source required")
    file = source / "utils/agentreward.py"
    original = file.read_bytes()
    content = original.decode("utf-8")
    edits = [("reward = reward + (-1.5)", f"reward = reward + ({false_negative})"),
             ("reward = reward + (-0.6)", f"reward = reward + ({false_positive})")]
    for old, new in edits:
        if content.count(old) != 1:
            raise ValueError("source version does not match reward patch")
        content = content.replace(old, new)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    if not include_slow_learner:
        runner = destination / "exp/opt_rl2_anomaly.py"
        runner_text = runner.read_text(encoding="utf8")
        old = 'model_list = [checkpoint for checkpoint in sorted(os.listdir(model_path))]'
        new = 'model_list = [checkpoint for checkpoint in sorted(os.listdir(model_path)) if "slow_learner" not in checkpoint]'
        if runner_text.count(old) != 1:
            raise ValueError("source version does not match slow-learner ablation")
        runner.write_text(runner_text.replace(old, new), encoding="utf8")
    corrected = content.encode("utf-8")
    (destination / "utils/agentreward.py").write_bytes(corrected)
    manifest = {"file": "utils/agentreward.py", "source": str(source), "corrected": str(destination),
                "original_sha256": hashlib.sha256(original).hexdigest(),
                "corrected_sha256": hashlib.sha256(corrected).hexdigest(),
                "false_negative": false_negative, "false_positive": false_positive,
                "include_slow_learner": include_slow_learner,
                "boundary": "Author reward/checkpoint ablations generated; no training/evaluation acceptance."}
    if not include_slow_learner:
        manifest["runner_patch"] = {"file": "exp/opt_rl2_anomaly.py",
            "original_sha256": hashlib.sha256((source / "exp/opt_rl2_anomaly.py").read_bytes()).hexdigest(),
            "corrected_sha256": hashlib.sha256((destination / "exp/opt_rl2_anomaly.py").read_bytes()).hexdigest()}
    (destination / "local_patch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
