"""Compare completed strict runs without selecting or changing the default mode."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def write_scale_comparison(raw_dir: Path, experimental_dir: Path, output: Path):
    modes = ("raw_equation", "initial_reference_experimental")
    summaries, configs = [], []
    for directory, mode in zip((raw_dir, experimental_dir), modes):
        summary = pd.read_csv(directory / "metrics_summary.csv").iloc[0]
        config = json.loads((directory / "paper_run_config.json").read_text(encoding="utf-8"))
        if summary["scheduler_mode"] != "paper_strict" or summary["paper_objective_scale_mode"] != mode:
            raise ValueError(f"Unexpected scheduler or objective mode in {directory}")
        if tuple(int(summary[k]) for k in ("NP", "Ngen_max_stage1", "Ngen_max_stage2")) != (50, 200, 200):
            raise ValueError("Comparison requires normal 50/200/200 runs")
        config["optimization"].pop("paper_objective_scale_mode")
        summaries.append(summary)
        configs.append(config)
    if configs[0] != configs[1] or summaries[0]["seed"] != summaries[1]["seed"]:
        raise ValueError("Runs must have identical seed and configuration apart from scale mode")
    fingerprints = {}
    for name in ("initial_plans.pkl", "risk_map.npy", "conflicts_uncertain.csv", "key_flights.csv"):
        hashes = [hashlib.sha256((d / name).read_bytes()).hexdigest() for d in (raw_dir, experimental_dir)]
        if hashes[0] != hashes[1]:
            raise ValueError(f"Initial inputs differ: {name}")
        fingerprints[name] = hashes[0]
    rows = []
    for mode, s in zip(modes, summaries):
        rows.append(dict(mode=mode, stage1_Nc=s["stage1_final_conflict_points"],
                         final_Nc=s["stage2_final_conflict_points"], Tdelay=s["final_Tdelay"],
                         Tair=s["final_Tair"], ORISK=s["final_ORISK"], n_delay=s["final_n_delay"],
                         changed_flights=s["changed_flight_count_two_stage"],
                         fitness=s["final_paper_fitness"], runtime=s["total_runtime"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    lines = ["# Objective scale comparison", "", "Default remains raw_equation. No automatic selection.",
             "Fitness values have different scales and must not be compared directly.",
             "Identical initial inputs verified by SHA256; seed and all other configuration match.", "",
             "Runtime is observed wall time, not a controlled performance benchmark.", "",
             "```text", frame.to_string(index=False), "```", "", "## Initial input fingerprints", ""]
    lines.extend(f"- {name}: {value}" for name, value in fingerprints.items())
    from src.paper_consistency import qualitative_observations, stage1_coverage_observation

    for mode, directory, summary in zip(modes, (raw_dir, experimental_dir), summaries):
        lines += ["", f"## {mode}", "", stage1_coverage_observation(directory, summary), ""]
        lines.extend("- " + value for value in qualitative_observations(summary))
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("outputs"))
    parser.add_argument("--experimental", type=Path, default=Path("outputs_initial_reference_experimental"))
    parser.add_argument("--output", type=Path, default=Path("outputs/objective_scale_comparison.csv"))
    args = parser.parse_args()
    print(write_scale_comparison(args.raw, args.experimental, args.output).to_string(index=False))


if __name__ == "__main__":
    main()
