"""Evaluation metrics used by SAGE scripts."""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

from sage.rewards import get_polyline_yaw


def get_kinematic_profiles(trajectory: np.ndarray) -> dict[str, np.ndarray]:
    if len(trajectory) < 3:
        return {"vel": np.array([]), "acc": np.array([]), "yaw_rate": np.array([])}

    dt = 0.1
    positions = trajectory[:, :2]
    velocities = np.gradient(positions, dt, axis=0, edge_order=2)
    speeds = np.linalg.norm(velocities, axis=1)
    accelerations = np.gradient(velocities, dt, axis=0, edge_order=2)
    acceleration_magnitude = np.linalg.norm(accelerations, axis=1)
    headings = get_polyline_yaw(positions)
    angular_velocities = np.gradient(np.unwrap(headings), dt, edge_order=2)
    return {"vel": speeds, "acc": acceleration_magnitude, "yaw_rate": angular_velocities}


def calculate_distributional_realism(gen_traj: np.ndarray, gt_traj: np.ndarray) -> dict[str, float]:
    gen_profiles = get_kinematic_profiles(gen_traj)
    gt_profiles = get_kinematic_profiles(gt_traj)

    wd_vel = 0.0
    if gen_profiles["vel"].size > 0 and gt_profiles["vel"].size > 0:
        wd_vel = wasserstein_distance(gen_profiles["vel"], gt_profiles["vel"])

    wd_acc = 0.0
    if gen_profiles["acc"].size > 0 and gt_profiles["acc"].size > 0:
        wd_acc = wasserstein_distance(gen_profiles["acc"], gt_profiles["acc"])

    wd_yaw_rate = 0.0
    if gen_profiles["yaw_rate"].size > 0 and gt_profiles["yaw_rate"].size > 0:
        wd_yaw_rate = wasserstein_distance(gen_profiles["yaw_rate"], gt_profiles["yaw_rate"])

    return {
        "wd_velocity": float(wd_vel),
        "wd_acceleration": float(wd_acc),
        "wd_yaw_rate": float(wd_yaw_rate),
        "realism_meta_metric": float((wd_vel + wd_acc + wd_yaw_rate) / 3.0),
    }
