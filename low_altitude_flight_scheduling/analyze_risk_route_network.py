"""A-D population/A* structure comparison with frozen building map and traffic."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.risk_route_network import run_structure_experiments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--environment-seed", type=int, default=None)
    parser.add_argument("--traffic-seed", type=int, default=None)
    parser.add_argument("--n-flights", type=int, default=100, help="Default 100; smaller values are smoke tests only")
    parser.add_argument("--beta", type=float, default=None, help="Explicit sensitivity value; never selected automatically")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cfg = load_config(root / args.config, {"optimization": {"scheduler_mode": "paper_strict"}})
    _, directory = run_structure_experiments(cfg, root / args.outputs, args.environment_seed, args.traffic_seed,
                                             args.n_flights, args.beta, root)
    print(f"Comparison outputs: {directory}")


if __name__ == "__main__":
    main()
