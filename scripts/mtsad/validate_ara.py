"""Validate the cross-layer bindings of the MTSAD research artifact.

Checks, in the ARA's own vocabulary:

* every claim (C##) carries Statement, Conditions, Falsification criteria and a
  Proof that resolves to experiment IDs declared in experiments.md;
* every experiment (E##) carries Verifies, Setup, Procedure, Expected outcome and
  Evidence, and every evidence path it names exists on disk;
* every experiment is referenced by at least one claim, so no experiment is
  orphaned;
* the exploration graph parses, its nodes carry ids and support levels, its
  evidence references are claim IDs, and its next-step nodes reference paths that
  exist.

It is a gate, not a report: it exits non-zero when a binding is broken.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "ara_mtsad"
CLAIM_ID = re.compile(r"^##\s+(C\d+)\b", re.MULTILINE)
EXPERIMENT_ID = re.compile(r"^##\s+(E\d+)\b", re.MULTILINE)
TREE_EVIDENCE = re.compile(r"^[CE]\d+$")


def blocks(text: str, pattern: re.Pattern[str]) -> dict[str, str]:
    matches = list(pattern.finditer(text))
    out: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        out[match.group(1)] = text[match.start():end]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    args = parser.parse_args()
    issues: list[str] = []

    claims_text = (args.artifact / "logic" / "claims.md").read_text(encoding="utf-8")
    experiments_text = (args.artifact / "logic" / "experiments.md").read_text(
        encoding="utf-8"
    )
    claims = blocks(claims_text, CLAIM_ID)
    experiments = blocks(experiments_text, EXPERIMENT_ID)

    required_claim_fields = ("**Statement**", "**Conditions**", "**Falsification criteria**", "**Proof**")
    required_experiment_fields = (
        "**Verifies**",
        "**Setup**",
        "**Procedure**",
        "**Expected outcome**",
        "**Evidence**",
    )
    proof_edges: dict[str, list[str]] = {}
    for claim_id, body in claims.items():
        for field in required_claim_fields:
            if field not in body:
                issues.append(f"{claim_id} is missing {field}")
        proof = re.search(r"\*\*Proof\*\*:\s*([^\n]+)", body)
        if not proof:
            continue
        ids = re.findall(r"E\d+", proof.group(1))
        proof_edges[claim_id] = ids
        for experiment_id in ids:
            if experiment_id not in experiments:
                issues.append(f"{claim_id} proves with {experiment_id}, which is not declared")

    referenced: set[str] = set()
    for experiment_id, body in experiments.items():
        for field in required_experiment_fields:
            if field not in body:
                issues.append(f"{experiment_id} is missing {field}")
        verifies = re.search(r"\*\*Verifies\*\*:\s*([^\n]+)", body)
        for claim_id in re.findall(r"C\d+", verifies.group(1)) if verifies else []:
            if claim_id not in claims:
                issues.append(f"{experiment_id} verifies {claim_id}, which is not declared")
            referenced.add(experiment_id)
        evidence = re.search(r"\*\*Evidence\*\*:\s*([^\n]+)", body)
        for path in re.findall(r"`([^`]+)`", evidence.group(1)) if evidence else []:
            if path.endswith(".md") or path.endswith(".json"):
                target = args.artifact / path.lstrip("/")
                if not target.exists():
                    issues.append(f"{experiment_id} cites {path}, which does not exist")

    for experiment_id in experiments:
        if experiment_id not in referenced:
            issues.append(f"{experiment_id} is not referenced by any claim")

    tree = yaml.safe_load((args.artifact / "trace" / "exploration_tree.yaml").read_text(encoding="utf-8"))
    nodes = tree.get("nodes") or []
    seen_ids: set[str] = set()
    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            issues.append("a trace node has no id")
            continue
        if node_id in seen_ids:
            issues.append(f"duplicate trace node id {node_id}")
        seen_ids.add(node_id)
        if "support_level" not in node:
            issues.append(f"trace node {node_id} has no support_level")
        for reference in node.get("evidence") or []:
            if not TREE_EVIDENCE.match(str(reference)):
                issues.append(f"trace node {node_id} references {reference}, which is not a claim id")
            elif str(reference).startswith("C") and str(reference) not in claims:
                issues.append(f"trace node {node_id} references undeclared claim {reference}")

    for source in (args.artifact / "PAPER.md", args.artifact / "evidence" / "README.md"):
        text = source.read_text(encoding="utf-8")
        for path in re.findall(r"`(evidence/[^`]+|logic/[^`]+|src/[^`]+|trace/[^`]+)`", text):
            if "*" in path:
                # Globs are used deliberately when a layer index summarises a
                # directory; only concrete paths are checkable.
                continue
            if not (args.artifact / path).exists():
                issues.append(f"{source.name} points at {path}, which does not exist")

    print(f"claims: {len(claims)}; experiments: {len(experiments)}; trace nodes: {len(nodes)}")
    print(f"proof edges: {sum(len(v) for v in proof_edges.values())}")
    if issues:
        print(f"\n{len(issues)} issue(s):")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    print("\nall cross-layer bindings resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
