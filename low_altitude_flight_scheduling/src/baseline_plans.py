"""Immutable initial-flight-plan baseline used by formal reproduction runs."""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path


BASELINE_SOURCE_COMMIT = "11e1a049d5f397362a97d7620ecaa4553209a6d4"
BASELINE_SHA256 = "5580bd2ad2cbdcea7dac0b3f8ac8dd990babce979aee4b274285986bc68facf7"
BASELINE_RELATIVE_PATH = Path("data") / "baseline_initial_plans.pkl"


def baseline_path(project_root: str | Path) -> Path:
    return Path(project_root) / BASELINE_RELATIVE_PATH


def baseline_plan_hash(project_root: str | Path) -> str:
    path = baseline_path(project_root)
    if not path.exists():
        raise FileNotFoundError(f"Formal baseline plan file is missing: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != BASELINE_SHA256:
        raise RuntimeError(
            f"baseline_plan_hash changed: expected {BASELINE_SHA256}, got {digest}. "
            "Refusing to use a different initial-flight-plan baseline."
        )
    return digest


def baseline_metadata(project_root: str | Path) -> dict[str, str | int]:
    return {
        "path": str(baseline_path(project_root)),
        "sha256": baseline_plan_hash(project_root),
        "source_commit": BASELINE_SOURCE_COMMIT,
        "flight_count": 100,
    }


def load_baseline_plans(project_root: str | Path):
    """Load the byte-validated historical plans without regenerating any route."""
    path = baseline_path(project_root)
    baseline_plan_hash(project_root)
    plans = pickle.loads(path.read_bytes())
    if len(plans) != 100:
        raise RuntimeError(f"Baseline must contain 100 flights, got {len(plans)}")
    if [plan.id for plan in plans] != list(range(100)):
        raise RuntimeError("Baseline flight IDs must be exactly 0..99")
    return plans
