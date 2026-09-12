"""Compare two completed child runs; never launch runs or choose a default mode."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.run_archive import validate_run_id


def write_scale_comparison(raw_dir, experimental_dir, output_root, comparison_id=None):
    directories = [Path(raw_dir), Path(experimental_dir)]
    modes = ("raw_equation", "initial_reference_experimental")
    summaries, configs, manifests = [], [], []
    for directory, mode in zip(directories, modes):
        s = pd.read_csv(directory / "metrics_summary.csv").iloc[0]
        manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        cfg = json.loads((directory / "paper_run_config.json").read_text(encoding="utf-8"))
        if (manifest["status"] != "completed" or s["scheduler_mode"] != "paper_strict"
                or s["paper_objective_scale_mode"] != mode or manifest["objective_scale_mode"] != mode
                or s["run_id"] != manifest["run_id"]):
            raise ValueError(f"Not a completed strict {mode} run: {directory}")
        budget = tuple(int(s[k]) for k in ("NP", "Ngen_max_stage1", "Ngen_max_stage2"))
        if budget != ((20,50,50) if manifest["quick"] else (50,200,200)):
            raise ValueError("Run budget does not match its quick/normal manifest")
        cfg["optimization"].pop("paper_objective_scale_mode")
        cfg.pop("run", None)
        summaries.append(s)
        configs.append(cfg)
        manifests.append(manifest)
    if (configs[0] != configs[1] or summaries[0]["seed"] != summaries[1]["seed"]
            or manifests[0]["quick"] != manifests[1]["quick"]):
        raise ValueError("Runs require identical seed, budget and configuration apart from scale mode")
    if summaries[0].run_id == summaries[1].run_id:
        raise ValueError("Comparison requires two distinct child run_ids")
    fingerprints = {}
    for name in ("initial_plans.pkl", "risk_map.npy", "conflicts_uncertain.csv", "key_flights.csv"):
        hashes = []
        for d in directories:
            if name.endswith(".csv"):
                frame = pd.read_csv(d/name).drop(columns="run_id", errors="ignore")
                value = frame.to_csv(index=False).encode("utf-8")
            else:
                value = (d/name).read_bytes()
            hashes.append(hashlib.sha256(value).hexdigest())
        if hashes[0] != hashes[1]:
            raise ValueError(f"Initial inputs differ: {name}")
        fingerprints[name] = hashes[0]
    comparison_id = comparison_id or f"cmp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}_seed{int(summaries[0].seed)}"
    validate_run_id(comparison_id)
    output = Path(output_root) / "comparisons" / comparison_id
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for mode, s in zip(modes, summaries):
        rows.append(dict(run_id=s.run_id, comparison_id=comparison_id, objective_mode=mode,
                         initial_Nc=s["stage1_initial_conflict_points"], stage1_Nc=s["stage1_final_conflict_points"],
                         final_Nc=s["stage2_final_conflict_points"], Tdelay=s["final_Tdelay"], Tair=s["final_Tair"],
                         ORISK=s["final_ORISK"], n_delay=s["final_n_delay"], changed_flights=s["changed_flight_count_two_stage"],
                         fitness=s["final_paper_fitness"], runtime=s["total_runtime"]))
    frame = pd.DataFrame(rows)
    frame.to_csv(output/"objective_scale_comparison.csv", index=False)
    comparison_manifest = dict(comparison_id=comparison_id, created_at=datetime.now().astimezone().isoformat(),
                               child_runs=[m["run_id"] for m in manifests], child_directories=[str(d.resolve()) for d in directories],
                               seed=int(summaries[0].seed), quick=manifests[0]["quick"], input_fingerprints=fingerprints,
                               default_mode="raw_equation", automatic_mode_selection=False,
                               note="CSV input fingerprints exclude run_id; fitness scales are not directly comparable.")
    (output/"comparison_manifest.json").write_text(json.dumps(comparison_manifest, indent=2), encoding="utf-8")
    (output/"objective_scale_comparison.md").write_text("# Objective scale comparison\n\nDefault remains raw_equation; no automatic mode selection.\n"
        "Fitness scales cannot be compared directly. Runtime is observed wall time, not a controlled benchmark.\n\n```text\n"
        + frame.to_string(index=False) + "\n```\n", encoding="utf-8")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--raw-run-id", "--raw", dest="raw", required=True, type=Path)
    parser.add_argument("--norm-run-id", "--experimental", dest="experimental", required=True, type=Path)
    parser.add_argument("--comparison-id", default=None)
    args = parser.parse_args()
    directories = [p if (p/"run_manifest.json").exists() else args.outputs/"runs"/p for p in (args.raw,args.experimental)]
    print(write_scale_comparison(*directories,args.outputs,args.comparison_id).to_string(index=False))


if __name__ == "__main__":
    main()
