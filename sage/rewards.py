"""Reward and penalty functions shared by SAGE training and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from shapely.geometry import LineString, Polygon, box
from shapely.strtree import STRtree


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    interval = np.pad(data, window_size // 2, "edge")
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, "valid")


def get_polyline_yaw(polyline: np.ndarray) -> np.ndarray:
    polyline = np.asarray(polyline)
    if polyline.shape[0] < 2:
        return np.zeros(polyline.shape[0])
    polyline_post = np.roll(polyline, shift=-1, axis=0)
    diff = polyline_post - polyline
    polyline_yaw = np.arctan2(diff[:, 1], diff[:, 0])
    polyline_yaw[-1] = polyline_yaw[-2]
    for i in range(len(polyline_yaw) - 1):
        if polyline_yaw[i + 1] - polyline_yaw[i] > 1.5 * np.pi:
            polyline_yaw[i + 1] -= 2 * np.pi
        elif polyline_yaw[i] - polyline_yaw[i + 1] > 1.5 * np.pi:
            polyline_yaw[i + 1] += 2 * np.pi
    return moving_average(polyline_yaw, window_size=5)


def get_polyline_vel(polyline: np.ndarray, dt: float = 0.1) -> np.ndarray:
    polyline = np.asarray(polyline)
    polyline_post = np.roll(polyline, shift=-1, axis=0)
    polyline_post[-1] = polyline[-1]
    return (polyline_post - polyline) / dt


def intersect(line_a: np.ndarray, line_b: np.ndarray) -> bool:
    v1 = (line_a[0] - line_b[0], line_a[1] - line_b[1])
    v2 = (line_a[0] - line_b[2], line_a[1] - line_b[3])
    v0 = (line_a[0] - line_a[2], line_a[1] - line_a[3])
    a = v0[0] * v1[1] - v0[1] * v1[0]
    b = v0[0] * v2[1] - v0[1] * v2[0]

    line_a, line_b = line_b, line_a
    v1 = (line_a[0] - line_b[0], line_a[1] - line_b[1])
    v2 = (line_a[0] - line_b[2], line_a[1] - line_b[3])
    v0 = (line_a[0] - line_a[2], line_a[1] - line_a[3])
    c = v0[0] * v1[1] - v0[1] * v1[0]
    d = v0[0] * v2[1] - v0[1] * v2[0]
    return a * b < 0 and c * d < 0


def _vectorized_get_corners(
    pos: np.ndarray, yaw: np.ndarray, length: np.ndarray | float, width: np.ndarray | float
) -> np.ndarray:
    pos = np.asarray(pos)
    yaw = np.asarray(yaw)
    length = np.asarray(length)
    width = np.asarray(width)

    if yaw.ndim < pos.ndim - 1:
        yaw = np.broadcast_to(yaw, pos.shape[:-1])
    if length.ndim < pos.ndim - 1:
        length = np.broadcast_to(length, pos.shape[:-1])
    if width.ndim < pos.ndim - 1:
        width = np.broadcast_to(width, pos.shape[:-1])

    half_l = length / 2
    half_w = width / 2
    offsets = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * np.stack(
        [half_l, half_w], axis=-1
    )[..., np.newaxis, :]
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    rot_matrix = np.stack(
        [np.stack([cos_y, -sin_y], axis=-1), np.stack([sin_y, cos_y], axis=-1)],
        axis=-2,
    )
    rotated_offsets = offsets[..., np.newaxis, :] @ rot_matrix[..., np.newaxis, :, :]
    return pos[..., np.newaxis, :] + np.squeeze(rotated_offsets, axis=-2)


def _query_tree_geometries(tree: STRtree, geometries: list[LineString], query_geom: Polygon) -> list[LineString]:
    results = tree.query(query_geom)
    if len(results) == 0:
        return []
    first = results[0]
    if isinstance(first, (int, np.integer)):
        return [geometries[int(idx)] for idx in results]
    return list(results)


def calculate_map_violation_penalty(
    trajectory: np.ndarray,
    map_features: dict,
    traffic_motion_feat: dict,
    adv_info: dict,
    cross_solid_line_scale: float = 50.0,
    crash_object_scale: float = 10.0,
) -> dict[str, float]:
    penalties = {
        "cross_solid_line_penalty": 0.0,
        "crash_object_penalty": 0.0,
    }

    impassable_walls = []
    impassable_line_types = {"ROAD_EDGE_BOUNDARY"}
    for feature_data in map_features.values():
        if feature_data.get("type") in impassable_line_types and "polyline" in feature_data:
            polyline = np.asarray(feature_data["polyline"])[:, :2]
            if len(polyline) >= 2:
                impassable_walls.append(LineString(polyline))

    wall_tree = STRtree(impassable_walls) if impassable_walls else None
    yaw_adv_traj = get_polyline_yaw(trajectory[:, :2])
    width_adv, length_adv = np.array(adv_info["w"]), np.array(adv_info["l"])

    if wall_tree and len(trajectory) > 1:
        adv_all_corners = _vectorized_get_corners(
            trajectory[:, :2], yaw_adv_traj, length_adv, width_adv
        )
        for corners in adv_all_corners:
            adv_polygon = Polygon(corners)
            if any(
                adv_polygon.intersects(wall)
                for wall in _query_tree_geometries(wall_tree, impassable_walls, adv_polygon)
            ):
                penalties["cross_solid_line_penalty"] = cross_solid_line_scale
                break

    all_x = _to_numpy(traffic_motion_feat["state/future/x"])
    all_y = _to_numpy(traffic_motion_feat["state/future/y"])
    all_pos = np.stack([all_x, all_y], axis=-1)
    all_yaw = _to_numpy(traffic_motion_feat["state/future/bbox_yaw"])
    all_len = _to_numpy(traffic_motion_feat["state/future/length"])
    all_wid = _to_numpy(traffic_motion_feat["state/future/width"])
    all_valid = _to_numpy(traffic_motion_feat["state/future/valid"])

    num_timesteps = len(trajectory)
    valid_mask = all_valid[..., np.newaxis, np.newaxis] > 0
    all_corners = _vectorized_get_corners(all_pos, all_yaw, all_len, all_wid)

    all_corners_masked = np.where(valid_mask, all_corners, np.inf)
    min_coords = np.min(all_corners_masked, axis=(1, 2))
    all_corners_masked = np.where(valid_mask, all_corners, -np.inf)
    max_coords = np.max(all_corners_masked, axis=(1, 2))

    adv_traj_corners = _vectorized_get_corners(
        trajectory[:, :2], yaw_adv_traj, length_adv, width_adv
    )
    adv_min_coord = np.min(adv_traj_corners, axis=(0, 1))
    adv_max_coord = np.max(adv_traj_corners, axis=(0, 1))
    adv_trajectory_box = box(
        adv_min_coord[0], adv_min_coord[1], adv_max_coord[0], adv_max_coord[1]
    )

    potential_collision_indices = []
    for i in range(2, all_pos.shape[0]):
        if np.all(min_coords[i] == np.inf):
            continue
        other_box = box(min_coords[i, 0], min_coords[i, 1], max_coords[i, 0], max_coords[i, 1])
        if adv_trajectory_box.intersects(other_box):
            potential_collision_indices.append(i)

    if not potential_collision_indices:
        return penalties

    candidate_corners = all_corners[potential_collision_indices]
    for t in range(num_timesteps):
        adv_polygon = Polygon(adv_traj_corners[t])
        for i, original_idx in enumerate(potential_collision_indices):
            if all_valid[original_idx, t]:
                other_polygon = Polygon(candidate_corners[i, t])
                if adv_polygon.intersects(other_polygon):
                    penalties["crash_object_penalty"] = crash_object_scale
                    return penalties

    return penalties


def calculate_realism_penalty(trajectory: np.ndarray, adv_info: dict) -> dict[str, float]:
    if len(trajectory) < 5:
        return {"kinematic_penalty": 0.0, "behavior_penalty": 0.0}

    dt = 0.1
    positions = trajectory[:, :2]
    velocities = np.gradient(positions, dt, axis=0, edge_order=2)
    speeds = np.linalg.norm(velocities, axis=1)
    headings = get_polyline_yaw(positions)
    unwrapped_headings = np.unwrap(headings)
    angular_velocities = np.gradient(unwrapped_headings, dt, edge_order=2)

    longitudinal_accel = np.gradient(speeds, dt, edge_order=2)
    lateral_accel = speeds * angular_velocities

    accel_penalty = np.mean(np.log1p(np.exp(np.abs(longitudinal_accel) - 7.0)))
    lat_accel_penalty = np.mean(np.log1p(np.exp(np.abs(lateral_accel) - 6.0)))
    ang_vel_penalty = np.mean(np.log1p(np.exp(np.abs(angular_velocities) - 0.8)))
    kinematic_penalty = 5.0 * (accel_penalty + lat_accel_penalty) + 5.0 * ang_vel_penalty

    total_heading_change = np.abs(unwrapped_headings[-1] - unwrapped_headings[0])
    high_turn_penalty = 5.0 * np.log1p(np.exp(total_heading_change - np.pi))
    turn_while_slow_metric = np.abs(angular_velocities) / (speeds + 1e-2)
    stop_and_turn_penalty = 3.0 * np.mean(turn_while_slow_metric)

    return {
        "kinematic_penalty": float(kinematic_penalty),
        "behavior_penalty": float(high_turn_penalty + stop_and_turn_penalty),
    }


def calculate_adversarial_reward(
    adv_traj: np.ndarray, ego_traj: np.ndarray, adv_info: dict, ego_info: dict
) -> tuple[float, bool]:
    common_len = min(len(adv_traj), len(ego_traj))
    if common_len < 2:
        return 0.0, False

    adv_traj_aligned = adv_traj[:common_len]
    ego_traj_aligned = ego_traj[:common_len]
    yaw_adv = get_polyline_yaw(adv_traj_aligned).reshape(-1, 1)
    yaw_ego = get_polyline_yaw(ego_traj_aligned).reshape(-1, 1)

    def get_bbox(traj: np.ndarray, yaw: np.ndarray, width: float, length: float):
        cos_theta, sin_theta = np.cos(yaw), np.sin(yaw)
        p1 = traj + np.hstack(
            [0.5 * length * cos_theta + 0.5 * width * sin_theta,
             0.5 * length * sin_theta - 0.5 * width * cos_theta]
        )
        p2 = traj + np.hstack(
            [0.5 * length * cos_theta - 0.5 * width * sin_theta,
             0.5 * length * sin_theta + 0.5 * width * cos_theta]
        )
        p3 = traj + np.hstack(
            [-0.5 * length * cos_theta - 0.5 * width * sin_theta,
             -0.5 * length * sin_theta + 0.5 * width * cos_theta]
        )
        p4 = traj + np.hstack(
            [-0.5 * length * cos_theta + 0.5 * width * sin_theta,
             -0.5 * length * sin_theta - 0.5 * width * cos_theta]
        )
        return p1, p2, p3, p4

    adv_corners = get_bbox(adv_traj_aligned, yaw_adv, adv_info["w"], adv_info["l"])
    ego_corners = get_bbox(ego_traj_aligned, yaw_ego, ego_info["w"], ego_info["l"])

    for t in range(common_len):
        adv_points = [corner[t] for corner in adv_corners]
        ego_points = [corner[t] for corner in ego_corners]
        if Polygon(adv_points).intersects(Polygon(ego_points)):
            collision_reward = 10.0 * (1.0 - t / common_len)
            return float(collision_reward), True
        adv_edges = [(adv_points[i], adv_points[(i + 1) % 4]) for i in range(4)]
        ego_edges = [(ego_points[i], ego_points[(i + 1) % 4]) for i in range(4)]
        for adv_edge in adv_edges:
            for ego_edge in ego_edges:
                if intersect(np.concatenate(adv_edge), np.concatenate(ego_edge)):
                    collision_reward = 10.0 * (1.0 - t / common_len)
                    return float(collision_reward), True

    distances = np.linalg.norm(adv_traj_aligned - ego_traj_aligned, axis=1)
    min_dist = np.min(distances)
    proximity_reward = np.exp(-0.2 * min_dist) if min_dist <= 20 else 0.0
    return float(proximity_reward), False
