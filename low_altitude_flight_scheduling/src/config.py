from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any

from .utils import deep_update


DEFAULT_CONFIG: dict[str, Any] = {
    "airspace": {
        "physical_size": [6000, 6000, 120],
        "grid_cell_size": [100, 100, 30],
        "grid_shape": [60, 60, 4],
        "obstacle_ratio": 0.10,
    },
    "flight": {
        "n_flights": 100,
        "takeoff_time_window": [0, 1800],
        "default_speed": 10.0,
        "route_distance_target": 6000,
        "random_seed": 2025,
        "min_flight_layer": 1,
        "altitude_preference_weights": [0.30, 0.45, 0.25],
        "altitude_preference_strength": 0.30,
        "route_random_bias_strength": 0.12,
        "traffic_pulse_count": 6,
        "traffic_pulse_jitter": 130,
    },
        "flight_generation": {
            "mode": "heterogeneous_random",
            "n_flights": 100,
            "distance_target_m": 6000,
            "distance_tolerance_m": 1500,
            "max_sampling_attempts": 30000,
            "od_pattern": {
                "hotspot_count": 10,
                "hotspot_std_cells": 6.5,
                "hub_to_hub_ratio": 0.45,
                "edge_to_edge_ratio": 0.25,
                "center_crossing_ratio": 0.15,
                "pure_random_ratio": 0.15,
                "center_bias_strength": 0.25,
            },
            "takeoff_time": {
                "mode": "mixture_peaks",
                "window": [0, 1800],
                "peak_centers": [350, 800, 1250],
                "peak_stds": [150, 220, 180],
                "peak_weights": [0.25, 0.50, 0.25],
                "uniform_noise_ratio": 0.20,
            },
            "speed": {
                "mode": "heterogeneous",
                "mean": 10.0,
                "std": 1.2,
                "min": 7.0,
                "max": 14.0,
            },
            "altitude": {
                "ground_level": 0,
                "force_ground_start_goal": True,
                "cruise_level_probs": [0.05, 0.35, 0.45, 0.15],
                "force_cruise_altitude": True,
                "min_cruise_fraction": 0.65,
                "climb_descent_cells": 3,
            },
            "route_overlap": {
                "encourage_overlap": True,
                "shared_corridor_ratio": 0.12,
                "corridor_width_cells": 6,
                "corridor_count": 4,
                "use_multiple_corridors": True,
            },
        },
    "astar": {
        "vertical_move_penalty": 1.8,
        "altitude_preference_enabled": True,
        "cruise_altitude_penalty_weight": 0.35,
        "ground_hugging_penalty_weight": 0.8,
        "min_preferred_level": 1,
        "max_preferred_level": 3,
        "risk_weight": 0.65,
        "distance_weight": 0.35,
    },
    "conflict": {
        "t_conflict": 20.0,
        "cell_occupancy_time": 0.0,
        "alpha": 0.05,
        "sigma0": 1.0,
        "sigma_rate": 0.010,
        "conflict_spatial_buffer_cells": 0,
    },
    "optimization": {
        "scheduler_mode": "paper_strict",
        "paper_objective_scale_mode": "raw_equation",
        "n_jobs": 8,
        "legacy_stage1_key_ratio": 0.15,
        "legacy_stage1_key_selection_mode": "coverage_adaptive",
        "legacy_NP": 40,
        "x_range": [1, 60],
        "y_range": [1, 60],
        "z_range": [1, 4],
        "speed_range": [5, 20],
        "t_ATD_range": [1, 3600],
        "t_delay_max": 1800,
            "t_battery": 1200,
            "vmax": 20,
            "important_ratio": 0.10,
            "conflict_hard_penalty": 1_000_000,
            "safe_mode": True,
            "quick_disable_reroute": True,
            "max_runtime_seconds": 180,
            "quick_max_runtime_seconds": 60,
            "max_local_reroute_attempts": 20,
            "accept_only_if_global_conflicts_decrease": True,
            "stage1_key_ratio": 0.15,
            "stage1_key_selection_mode": "coverage_adaptive",
            "stage1_conflict_coverage_target": 0.82,
            "stage1_max_key_ratio": 0.25,
            "stage1_decision_mode": "continuous_atd_speed",
            "stage1_use_full_conflict_objective": True,
            "stage1_atd_range": [1, 3600],
            "stage1_respect_delay_max": True,
            "stage1_max_advance_seconds": 600,
            "stage1_conflict_point_penalty": 1_000_000,
            "stage1_conflict_pair_penalty": 200_000,
            "stage1_greedy_rounds": 50,
            "final_greedy_rounds": 30,
            "stage1_repair_rounds": 20,
            "stage2_repair_rounds": 20,
            "stage2_strategy": "independent_matching",
            "independent_matching_rounds": 60,
            "independent_matching_lrate": 0.5,
            "independent_matching_segment_limit": 24,
            "independent_matching_candidates_per_strategy": 20,
            "independent_matching_first_accepted_candidate": True,
            "independent_matching_first_pair_resolution": True,
            "independent_matching_first_strategy_success": True,
            "independent_matching_allow_reroute": True,
            "head_to_head_dot_threshold": -0.35,
            "stage2_cluster_points_threshold": 3,
            "stage2_cluster_neighbor_radius": 2,
            "independent_matching_strategy_priors": {
                "head_to_head": [0.15, 0.15, 0.70],
                "cluster": [0.50, 0.30, 0.20],
                "crossing": [0.25, 0.60, 0.15],
            },
            "local_reroute_windows": [8, 12],
            "local_reroute_radii": [1, 2],
            "delay_candidates": [-600, -420, -300, -240, -180, -120, -90, -60, -30, 0, 30, 60, 90, 120, 180, 240, 300, 420, 600],
            "speed_factors": [0.90, 0.95, 1.00, 1.05, 1.10, 1.15],
            "safety_margin_seconds": 10,
            "max_changed_flight_ratio": 0.10,
            "repair_segment_scan_limit": 8,
            "repair_action_limit": 40,
            "repair_first_improvement": True,
            "force_resolve_conflict_limit": 8,
            "force_allow_extra_delay_conflicts": 2,
            "force_shift_candidates": [-300, -240, -180, -120, -90, -60, -30, 30, 60, 90, 120, 180, 240, 300, 420, 600],
            "delay_count_threshold": 30,
            "delay_count_cap": 3,
        "greedy_candidate_limit": 6,
        "greedy_reroute_attempts": 3,
        "quick_repair_rounds": 120,
    },
    "risk": {
        "v_wind": 7.0,
        "wind_direction": 0.5,
        "S": 0.5,
        "r_buf": 0.3,
        "r_UAV": 0.2,
        "m": 1.4,
        "A_p": 1.0,
        "C_D_b": 1.3,
        "C_D_p": 0.7,
        "alpha_r": 0.8,
        "alpha_L": 0.2,
        "population_density_base": 3500,
        "failure_probability": 0.0001,
    },
    "fata": {
        "Parf": 0.2,
        "NP": 50,
        "Ngen_max_stage1": 200,
        "Ngen_max_stage2": 200,
        "Ngen_max": 120,
        "omega_c": 0.8,
        "omega_d": 0.25,
        "omega_r": 0.5,
        "omega_t": 0.25,
        "gamma": 5,
        "comparison_NP": 12,
        "comparison_Ngen_max": 25,
        "comparison_improved_restarts": 2,
        "quick_repeats": 3,
    },
    "adm": {"learning_rate": 0.5, "dominant_fraction": 0.20},
    "paper_encoding": {"local_window_segments": 4},
    "network": {"ci_l": 2, "top_k": 10},
    "visualization": {
        "show_ground_start_goal": True,
        "ground_z_m": 0.0,
        "airspace_cell_center_z": True,
    },
}


def _parse_scalar(text: str) -> Any:
    text = text.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() in {"null", "none"}:
        return None
    if text.startswith("[") and text.endswith("]"):
        return ast.literal_eval(text)
    try:
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)
    except ValueError:
        return text.strip("'\"")


def _minimal_yaml_load(raw: str) -> dict[str, Any]:
    """A tiny YAML subset parser for this config file when PyYAML is absent."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def load_config(path: str | Path = "config.yaml", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    config_path = Path(path)
    if config_path.exists():
        raw = config_path.read_text(encoding="utf-8")
        try:
            import yaml

            loaded = yaml.safe_load(raw) or {}
        except Exception:
            loaded = _minimal_yaml_load(raw)
        cfg = deep_update(cfg, loaded)
    if overrides:
        cfg = deep_update(cfg, overrides)
    cfg["legacy_scene"] = deep_update({
        "flight_generation": copy.deepcopy(cfg["flight_generation"]),
        "astar": copy.deepcopy(cfg["astar"]),
        "conflict": copy.deepcopy(cfg["conflict"]),
    }, cfg.get("legacy_scene", {}))
    if cfg["optimization"].get("scheduler_mode") == "paper_strict":
        scene = cfg["paper_scene"] = deep_update({
            "generation_mode": "paper_random", "n_flights": 100,
            "distance_target_m": 6000, "distance_tolerance_m": 1200,
            "takeoff": {"mode": "uniform", "min": 0, "max": 1800},
            "initial_speed": {"mode": "constant", "value": 10.0},
            "astar": {"risk_weight": 0.8, "distance_weight": 0.2,
                      "use_altitude_preference": False, "use_route_random_bias": False, "use_forced_corridor": False},
            "conflict": {"t_conflict": 30.0, "alpha": 0.05},
        }, cfg.get("paper_scene", {}))
        cfg.setdefault("scene_mode", "paper_strict_random")
        cfg["flight_generation"].update(
            mode=scene.get("generation_mode", "paper_random"),
            n_flights=scene.get("n_flights", 100),
            distance_target_m=scene.get("distance_target_m", 6000),
            distance_tolerance_m=scene.get("distance_tolerance_m", 1200),
        )
        cfg["flight"]["n_flights"] = cfg["flight_generation"]["n_flights"]
        cfg["conflict"].update(scene["conflict"])
        cfg["optimization"]["stage1_key_ratio"] = 0.10
        cfg["optimization"]["stage1_key_selection_mode"] = "fixed_ci"
    else:
        for section in ("flight_generation", "astar", "conflict"):
            cfg[section] = copy.deepcopy(cfg["legacy_scene"][section])
    return cfg


def apply_quick_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    if cfg["optimization"].get("scheduler_mode") == "paper_strict":
        cfg["fata"]["NP"] = 20
        cfg["fata"]["Ngen_max_stage1"] = 50
        cfg["fata"]["Ngen_max_stage2"] = 50
        return cfg
    cfg["fata"]["NP"] = min(int(cfg["fata"]["NP"]), 20)
    cfg["fata"]["Ngen_max"] = min(int(cfg["fata"]["Ngen_max"]), 50)
    cfg["fata"]["quick_repeats"] = 3
    cfg["optimization"]["quick_disable_reroute"] = True
    cfg["optimization"]["quick_max_runtime_seconds"] = min(float(cfg["optimization"].get("quick_max_runtime_seconds", 60)), 60)
    cfg["optimization"]["stage1_greedy_rounds"] = min(int(cfg["optimization"].get("stage1_greedy_rounds", 50)), 35)
    cfg["optimization"]["final_greedy_rounds"] = min(int(cfg["optimization"].get("final_greedy_rounds", 30)), 20)
    cfg["optimization"]["quick_repair_rounds"] = 40
    cfg["optimization"]["stage1_repair_rounds"] = min(int(cfg["optimization"].get("stage1_repair_rounds", 20)), 10)
    cfg["optimization"]["stage2_repair_rounds"] = min(int(cfg["optimization"].get("stage2_repair_rounds", 20)), 10)
    cfg["optimization"]["independent_matching_rounds"] = min(int(cfg["optimization"].get("independent_matching_rounds", 60)), 60)
    cfg["optimization"]["independent_matching_segment_limit"] = min(int(cfg["optimization"].get("independent_matching_segment_limit", 24)), 12)
    cfg["optimization"]["independent_matching_candidates_per_strategy"] = min(int(cfg["optimization"].get("independent_matching_candidates_per_strategy", 20)), 20)
    cfg["optimization"]["independent_matching_allow_reroute"] = True
    cfg["optimization"]["repair_segment_scan_limit"] = min(int(cfg["optimization"].get("repair_segment_scan_limit", 8)), 4)
    cfg["optimization"]["repair_action_limit"] = min(int(cfg["optimization"].get("repair_action_limit", 40)), 20)
    return cfg
