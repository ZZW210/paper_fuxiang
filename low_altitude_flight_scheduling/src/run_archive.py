"""Isolated, auditable outputs for every run_main invocation."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from threading import Lock

import pandas as pd
from matplotlib.figure import Figure

from .paper_performance import PerformanceCounters

_LAST_ID_TIME = None
_ID_LOCK = Lock()


def append_run_index(path, fields, row):
    lock_path = path.with_suffix(".lock")
    with lock_path.open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            with path.open("a", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                if fh.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def git_metadata(root):
    def git(*args):
        result = subprocess.run(["git", "-c", f"safe.directory={root.as_posix()}", *args], cwd=root,
                                capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else ""
    return git("rev-parse", "HEAD") or "unknown", bool(git("status", "--porcelain"))


def generate_run_id(mode, objective_mode, seed, commit="unknown"):
    global _LAST_ID_TIME
    with _ID_LOCK:
        now = datetime.now().astimezone()
        now = now.replace(microsecond=now.microsecond // 1000 * 1000)
        if _LAST_ID_TIME is not None and now <= _LAST_ID_TIME:
            now = _LAST_ID_TIME + timedelta(milliseconds=1)
        _LAST_ID_TIME = now
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return f"{timestamp}_{mode}_{objective_mode}_seed{seed}_{commit[:8]}"


def generate_network_run_id(population_mode, distance_scale, environment_seed, traffic_seed, commit="unknown"):
    timestamp = generate_run_id("network", "snapshot", traffic_seed, commit).split("_network_", 1)[0]
    value = (f"{timestamp}_network_{population_mode}_{distance_scale}_env{environment_seed}"
             f"_traffic{traffic_seed}_{commit[:8]}")
    validate_run_id(value)
    return value


def validate_run_id(run_id):
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,139}", run_id)
            or run_id.endswith(".") or run_id.split('.')[0].upper() in
            {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}):
        raise ValueError("run_id must be a Windows-safe filename, at most 140 characters")


class RunTee:
    def __init__(self, terminal, logfile, run_id):
        self.terminal, self.logfile, self.run_id = terminal, logfile, run_id
        self.line_start = True

    def write(self, value):
        for part in value.splitlines(keepends=True):
            prefix = f"[run_id={self.run_id}] " if self.line_start else ""
            self.terminal.write(prefix + part)
            self.logfile.write(prefix + part)
            self.line_start = part.endswith(('\n', '\r'))
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"


class RunArchive:
    def __init__(self, root, output_root, cfg, args):
        self.root, self.output_root = Path(root).resolve(), Path(output_root).resolve()
        self.cfg, self.args = cfg, args
        commit, dirty = git_metadata(self.root.parent)
        mode = cfg["optimization"]["scheduler_mode"]
        objective = cfg["optimization"].get("paper_objective_scale_mode", "raw_equation") if mode == "paper_strict" else "legacy"
        self.run_id = generate_run_id(mode, objective, args.seed, commit) if args.run_id is None else args.run_id
        validate_run_id(self.run_id)
        runs = (self.output_root / "runs").resolve()
        runs.mkdir(parents=True, exist_ok=True)
        self.directory = runs / self.run_id
        if args.run_id is None:
            while True:
                try:
                    self.directory.mkdir(exist_ok=False)
                    break
                except FileExistsError:
                    self.run_id = generate_run_id(mode, objective, args.seed, commit)
                    self.directory = runs / self.run_id
        elif self.directory.exists():
            if not args.overwrite_run:
                raise FileExistsError(f"Run already exists: {self.directory}")
            if self.directory.is_symlink() or self.directory.resolve().parent != runs:
                raise ValueError("Refusing to overwrite a run outside the archive root")
            shutil.rmtree(self.directory)
            self.directory.mkdir(exist_ok=False)
        else:
            self.directory.mkdir(exist_ok=False)
        for name in ("figures", "logs"):
            (self.directory / name).mkdir()
        self.profile = PerformanceCounters()
        self.start = time.perf_counter()
        self.manifest = dict(run_id=self.run_id, start_time=datetime.now().astimezone().isoformat(),
                             end_time=None, status="running", git_commit=commit, git_dirty=dirty,
                             seed=args.seed, environment_seed=cfg.get("environment_seed", args.seed),
                             traffic_seed=cfg.get("traffic_seed", args.seed),
                             scheduler_mode=mode, objective_scale_mode=objective, quick=args.quick,
                             NP_stage1=cfg["fata"]["NP"], NP_stage2=cfg["fata"]["NP"],
                             Ngen_stage1=cfg["fata"]["Ngen_max_stage1"], Ngen_stage2=cfg["fata"]["Ngen_max_stage2"],
                             n_jobs=cfg["optimization"]["n_jobs"], python_version=sys.version,
                             platform=platform.platform(), cpu_count=os.cpu_count(), config_file=str(self.root / args.config),
                             initial_conflicts=None, stage1_conflicts=None, final_conflicts=None,
                             blas_threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                             performance_timing_note="Inclusive worker CPU-wall timings overlap parent evaluation wall time; serialization is a pickle probe, not isolated IPC time.")
        cfg["run"] = dict(run_id=self.run_id, git_commit=commit, git_dirty=dirty,
                          output_directory=str(self.directory), quick=args.quick)
        self._write_snapshot()
        self._write_manifest()

    def _write_snapshot(self):
        try:
            import yaml
            text = yaml.safe_dump(self.cfg, sort_keys=False, allow_unicode=True)
        except ImportError:
            text = json.dumps(self.cfg, indent=2, ensure_ascii=False)
        (self.directory / "config_snapshot.yaml").write_text(text, encoding="utf-8")

    def _write_manifest(self):
        (self.directory / "run_manifest.json").write_text(json.dumps(self.manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    def __enter__(self):
        self.logfile = (self.directory / "stdout.log").open("w", encoding="utf-8")
        self.stdout, self.stderr = sys.stdout, sys.stderr
        sys.stdout = RunTee(self.stdout, self.logfile, self.run_id)
        sys.stderr = RunTee(self.stderr, self.logfile, self.run_id)
        self.savefig = Figure.savefig
        original, run_id, profile = self.savefig, self.run_id, self.profile
        def savefig(figure, filename, *args, **kwargs):
            tag = figure.text(0.01, 0.005, f"Run ID: {run_id}", fontsize=7)
            try:
                with profile.measure("file_output"):
                    return original(figure, filename, *args, **kwargs)
            finally:
                tag.remove()
        Figure.savefig = savefig
        self.to_csv = pd.DataFrame.to_csv
        original_csv, directory = self.to_csv, self.directory.resolve()
        def to_csv(frame, path_or_buf=None, *args, **kwargs):
            if isinstance(path_or_buf, (str, Path)) and Path(path_or_buf).resolve().is_relative_to(directory):
                if "run_id" in frame and not frame["run_id"].eq(run_id).all():
                    raise ValueError("CSV contains another run_id")
                frame = frame.drop(columns="run_id", errors="ignore").copy()
                frame.insert(0, "run_id", run_id)
                with profile.measure("file_output"):
                    return original_csv(frame, path_or_buf, *args, **kwargs)
            return original_csv(frame, path_or_buf, *args, **kwargs)
        pd.DataFrame.to_csv = to_csv
        return self

    def _finalize_outputs(self):
        summary_path = self.directory / "metrics_summary.csv"
        summary = pd.read_csv(summary_path).iloc[0].to_dict()
        elapsed = time.perf_counter() - self.start
        summary.update(run_id=self.run_id, runtime_total=elapsed, runtime_seconds=elapsed, total_runtime=elapsed)
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        self.manifest.update(initial_conflicts=summary.get("initial_conflicts_uncertain"),
                             stage1_conflicts=summary.get("final_conflicts_one_stage"),
                             final_conflicts=summary.get("final_conflicts_two_stage"))
        for filename in self.directory.rglob("*.csv"):
            frame = pd.read_csv(filename)
            if not frame.columns.size or frame.columns[0] != "run_id" or not frame.run_id.eq(self.run_id).all():
                frame = frame.drop(columns="run_id", errors="ignore")
                frame.insert(0, "run_id", self.run_id)
                frame.to_csv(filename, index=False)
        moved = []
        for filename in self.directory.glob("*.png"):
            filename.replace(self.directory / "figures" / filename.name)
            moved.append(filename.name)
        for filename in (*self.directory.glob("*.html"), *self.directory.glob("*.md")):
            content = filename.read_text(encoding="utf-8")
            for name in moved:
                content = content.replace(f"src='{name}'", f"src='figures/{name}'").replace(f'src="{name}"', f'src="figures/{name}"')
                content = content.replace(f"]({name})", f"](figures/{name})")
            filename.write_text(content, encoding="utf-8")
        final = self.directory / "final_plans_two_stage.pkl"
        if final.exists():
            shutil.copyfile(final, self.directory / "final_plans.pkl")
        if not (self.directory / "optimization_trace.csv").exists():
            shutil.copyfile(self.directory / "paper_convergence.csv", self.directory / "optimization_trace.csv")
        components = ("fata_population_evaluation", "fata_population_update", "conflict_detection", "objective",
                      "route_decode", "astar", "multiprocessing_serialization", "visualization", "file_output")
        profile_path = self.directory / "performance_profile.csv"
        previous = pd.read_csv(profile_path).set_index("component").to_dict("index") if profile_path.exists() else {}
        rows = []
        for component in components:
            calls = self.profile.values[component + "_calls"] + previous.get(component, {}).get("calls", 0)
            seconds = self.profile.values[component + "_seconds"] + previous.get(component, {}).get("total_seconds", 0)
            rows.append(dict(run_id=self.run_id, component=component, calls=calls, total_seconds=seconds,
                             mean_ms=1000 * seconds / calls if calls else 0, percentage=100 * seconds / elapsed))
        pd.DataFrame(rows).to_csv(profile_path, index=False)
        self._write_snapshot()
        summary = pd.read_csv(summary_path)
        population_assumptions = self.directory / "population_implementation_assumptions.md"
        if population_assumptions.exists():
            path = self.directory / "implementation_assumptions.md"
            text = path.read_text(encoding="utf-8") if path.exists() else "# Implementation assumptions\n"
            if "## Population and risk implementation assumptions" not in text:
                path.write_text(text + "\n" + population_assumptions.read_text(encoding="utf-8"), encoding="utf-8")
        elapsed = time.perf_counter() - self.start
        summary["runtime_total"] = summary["runtime_seconds"] = summary["total_runtime"] = elapsed
        summary.to_csv(summary_path, index=False)
        self.manifest["elapsed_seconds"] = elapsed

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc is None:
                try:
                    self._finalize_outputs()
                except Exception as error:
                    self.manifest.update(status="failed", error=str(error), end_time=datetime.now().astimezone().isoformat(),
                                         elapsed_seconds=time.perf_counter()-self.start)
                    traceback.print_exc()
                    self._write_manifest()
                    raise
                self.manifest["status"] = "completed"
            else:
                self.manifest.update(status="failed", error=str(exc))
                traceback.print_exception(exc_type, exc, tb)
                self.manifest["elapsed_seconds"] = time.perf_counter() - self.start
            self.manifest["end_time"] = datetime.now().astimezone().isoformat()
            self._write_manifest()
            if exc is None:
                fields = ("run_id", "timestamp", "seed", "mode", "objective_mode", "initial_conflicts", "stage1_conflicts", "final_conflicts", "runtime", "git_commit")
                row = dict(run_id=self.run_id, timestamp=self.manifest["end_time"], seed=self.args.seed,
                           mode=self.manifest["scheduler_mode"], objective_mode=self.manifest["objective_scale_mode"],
                           initial_conflicts=self.manifest["initial_conflicts"], stage1_conflicts=self.manifest["stage1_conflicts"],
                           final_conflicts=self.manifest["final_conflicts"], runtime=self.manifest["elapsed_seconds"], git_commit=self.manifest["git_commit"])
                index = self.output_root / "run_index.csv"
                append_run_index(index, fields, row)
                temporary = self.directory / "latest_run.tmp"
                temporary.write_text(self.run_id, encoding="utf-8")
                os.replace(temporary, self.output_root / "latest_run.txt")
                print("Run completed")
                print(f"Run ID: {self.run_id}")
                print(f"Output directory: {self.directory}")
        finally:
            Figure.savefig = self.savefig
            pd.DataFrame.to_csv = self.to_csv
            sys.stdout, sys.stderr = self.stdout, self.stderr
            self.logfile.close()
