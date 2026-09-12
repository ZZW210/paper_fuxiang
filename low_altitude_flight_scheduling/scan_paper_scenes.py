"""Scan complete random initial scenes without running FATA or ADM."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import time

import pandas as pd

from src.config import load_config
from src.run_archive import generate_run_id, git_metadata, validate_run_id
from src.scene_diagnostics import generate_and_analyze_scene, write_calibration_report


def scan_one(cfg, seed, scan_id, directory):
    row, _ = generate_and_analyze_scene(cfg, seed, f"{scan_id}_scene{seed}", Path(directory) / f"seed{seed}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=2020)
    parser.add_argument("--seed-end", type=int, default=2039, help="Inclusive final seed")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--environment-seed", type=int, default=None)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--outputs", default="outputs/scene_scans")
    parser.add_argument("--scan-id")
    parser.add_argument("--scene-mode", choices=["paper_strict_random", "paper_calibrated"], default="paper_strict_random")
    args = parser.parse_args()
    if args.seed_end < args.seed_start or args.n_jobs < 1:
        parser.error("Require seed-end >= seed-start and n-jobs >= 1")
    root = Path(__file__).resolve().parent
    cfg = load_config(root / args.config, {"optimization": {"scheduler_mode": "paper_strict"}})
    cfg["scene_mode"] = args.scene_mode
    if args.environment_seed is not None:
        cfg["environment_seed"] = args.environment_seed
    commit, dirty = git_metadata(root.parent)
    scan_id = args.scan_id or generate_run_id(args.scene_mode, "initial_network", args.seed_start, commit)
    validate_run_id(scan_id)
    directory = root / args.outputs / scan_id
    directory.mkdir(parents=True, exist_ok=False)
    seeds = list(range(args.seed_start, args.seed_end + 1))
    manifest = dict(scan_id=scan_id, scene_mode=args.scene_mode, seed_start=args.seed_start, seed_end=args.seed_end,
                    environment_seed=cfg["environment_seed"], traffic_seed_start=args.seed_start,
                    traffic_seed_end=args.seed_end, seed_scope="traffic_only",
                    seed_end_inclusive=True, candidate_count=len(seeds), n_jobs=args.n_jobs,
                    git_commit=commit, git_dirty=dirty, config=cfg, status="running",
                    optimizers_executed=False, start_time=datetime.now().astimezone().isoformat())
    path = directory / "scan_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started = time.perf_counter()
    rows = []
    try:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
            futures = {pool.submit(scan_one, cfg, seed, scan_id, str(directory)): seed for seed in seeds}
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                pd.DataFrame(rows).sort_values("seed").to_csv(directory / "scene_scan.csv", index=False)
                print(f"seed={row['seed']} det={row['deterministic_Nc']} unc={row['uncertain_Nc']} "
                      f"CI coverage={row['CI_top10_point_coverage']:.4f} runtime={row['runtime']:.2f}s", flush=True)
        if args.scene_mode == "paper_calibrated":
            best = write_calibration_report(directory, rows, args.seed_start, args.seed_end)
            manifest["selected_seed"] = best["seed"]
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest.update(runtime=time.perf_counter() - started, completed_candidates=len(rows),
                        end_time=datetime.now().astimezone().isoformat())
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"outputs: {directory}")


if __name__ == "__main__":
    main()
