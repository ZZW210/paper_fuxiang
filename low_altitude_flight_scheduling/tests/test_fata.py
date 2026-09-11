from __future__ import annotations

import numpy as np

from src.fata import fata_optimize, sphere_objective


def test_fata_improves_toy_objective() -> None:
    result = fata_optimize(sphere_objective, -5.0, 5.0, dim=5, population=12, max_iter=20, seed=3, improved=True)
    initial_like = float(np.sum(np.full(5, 5.0) ** 2))
    assert result.best_fitness < initial_like
    assert result.convergence[-1] <= result.convergence[0]


def test_fata_accepts_warm_start() -> None:
    optimum = np.array([1.5, -0.5])
    result = fata_optimize(
        lambda x: float(np.sum((x - optimum) ** 2)),
        -5.0,
        5.0,
        dim=2,
        population=4,
        max_iter=1,
        seed=3,
        initial_positions=optimum,
    )

    assert result.best_fitness == 0.0
    assert np.allclose(result.best_position, optimum)
