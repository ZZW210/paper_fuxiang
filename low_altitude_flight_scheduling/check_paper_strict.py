from __future__ import annotations

from pathlib import Path

from src.config import load_config
from src.paper_consistency import run_consistency_checks


def main():
    root = Path(__file__).resolve().parent
    checks = run_consistency_checks(load_config(root / "config.yaml"))
    for name, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    return 0 if all(passed for _, passed in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
