import argparse
import numpy as np
import os
import sys
import logging
from tqdm import trange, tqdm
from copy import deepcopy
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive_autopilot_safe import MetaDriveAutoPilotPolicy
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon, box
from shapely.strtree import STRtree


# ==========================================================================================
#  整合自 adv_generator.py 和 adv_utils.py 的核心代码
# ==========================================================================================
# 从 adv_generator.py 引入的辅助函数
def moving_average(data, window_size):
    interval = np.pad(data, window_size // 2, 'edge')
    window = np.ones(int(window_size)) / float(window_size)
    res = np.convolve(interval, window, 'valid')
    return res


def get_polyline_yaw(polyline):
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


def Intersect(l1, l2):
    v1 = (l1[0] - l2[0], l1[1] - l2[1])
    v2 = (l1[0] - l2[2], l1[1] - l2[3])
    v0 = (l1[0] - l1[2], l1[1] - l1[3])
    a = v0[0] * v1[1] - v0[1] * v1[0]
    b = v0[0] * v2[1] - v0[1] * v2[0]
    temp = l1
    l1 = l2
    l2 = temp
    v1 = (l1[0] - l2[0], l1[1] - l2[1])
    v2 = (l1[0] - l2[2], l1[1] - l2[3])
    v0 = (l1[0] - l1[2], l1[1] - l1[3])
    c = v0[0] * v1[1] - v0[1] * v1[0]
    d = v0[0] * v2[1] - v0[1] * v2[0]
    return a * b < 0 and c * d < 0


# DPOAdvGenerator, 整合自adv_generator.yp
from advgen.modeling.vectornet import VectorNet
import advgen.utils as advgen_utils
from advgen.adv_utils import process_data
from sage.splits import filter_ids_by_summary, scenario_ids


class DPOAdvGenerator(object):
    def __init__(self, parser, model_path='./advgen/pretrained/densetnt.bin'):
        advgen_utils.add_argument(parser)
        parser.set_defaults(
            other_params=['l1_loss', 'densetnt', 'goals_2D', 'enhance_global_graph', 'laneGCN', 'point_sub_graph',
                          'laneGCN-4', 'stride_10_2', 'raster', 'train_pair_interest'])
        parser.set_defaults(mode_num=32)
        parser.set_defaults(future_frame_num=80)
        args = parser.parse_args([])
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        logger = logging.getLogger(__name__)
        advgen_utils.init(args, logger)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_model = VectorNet(args).to(self.device)
        self.policy_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy_model.train()

        self.reference_model = VectorNet(args).to(self.device)
        self.reference_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self._internal_adv_generator = None

    def get_data_for_scenario(self, env):
        if not self._internal_adv_generator:
            _parser = argparse.ArgumentParser()
            _parser.add_argument('--OV_traj_num', type=int, default=32)
            _parser.add_argument('--AV_traj_num', type=int, default=1)
            from advgen.adv_generator import AdvGenerator as OriginalAdvGenerator
            self._internal_adv_generator = OriginalAdvGenerator(_parser)
        self._internal_adv_generator.before_episode(env)
        traffic_motion_feat = self._internal_adv_generator.storage[env.current_seed]['traffic_motion_feat']
        batch_data = process_data(traffic_motion_feat, self.args)
        adv_info = self._internal_adv_generator.storage[env.current_seed]['adv_info']
        ego_info = self._internal_adv_generator.storage[env.current_seed]['ego_info']
        ego_gt_future_traj = self._internal_adv_generator.storage[env.current_seed]['traffic_motion_feat'][
                                 'state/future/x'][0, :, np.newaxis]
        ego_gt_future_traj = np.concatenate([ego_gt_future_traj, self._internal_adv_generator.storage[env.current_seed][
                                                                     'traffic_motion_feat']['state/future/y'][0, :,
                                                                 np.newaxis]], axis=-1)
        adv_past_traj = self._internal_adv_generator.storage[env.current_seed]['adv_past']
        raw_map_features = env.engine.data_manager.get_scenario(env.current_seed)['map_features']
        return batch_data, traffic_motion_feat, adv_info, ego_info, ego_gt_future_traj, adv_past_traj, raw_map_features


def _vectorized_get_corners(pos: np.ndarray, yaw: np.ndarray, length: np.ndarray, width: np.ndarray) -> np.ndarray:
    """
    向量化计算包围盒角点 (V3 - 最终修正版, 简化且鲁棒)。
    Args:
        pos (np.ndarray): 位置数组, shape [..., 2]
        yaw (np.ndarray): 航向角数组, shape [...]
        length (np.ndarray): 长度数组, shape [...] or scalar
        width (np.ndarray): 宽度数组, shape [...] or scalar

    Returns:
        np.ndarray: 角点数组, shape [..., 4, 2]
    """
    # 确保输入是 NumPy 数组
    pos = np.asarray(pos)
    yaw = np.asarray(yaw)
    length = np.asarray(length)
    width = np.asarray(width)

    # 预处理：确保 yaw, length, width 可以广播到 pos 的批次维度
    # 例如, pos.shape = (80, 2), yaw.shape = (80,)
    # 我们希望 yaw, length, width 的形状都为 (80,)
    if yaw.ndim < pos.ndim - 1:
        yaw = np.broadcast_to(yaw, pos.shape[:-1])
    if length.ndim < pos.ndim - 1:
        length = np.broadcast_to(length, pos.shape[:-1])
    if width.ndim < pos.ndim - 1:
        width = np.broadcast_to(width, pos.shape[:-1])

    half_l = length / 2
    half_w = width / 2

    # 构建角点在车辆自身坐标系下的偏移量
    # shape: (..., 4, 2)
    # 我们通过在末尾添加新轴来确保广播正确
    s = np.stack([half_l, half_w], axis=-1)
    offsets = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * s[..., np.newaxis, :]

    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)

    # 构建旋转矩阵
    rot_matrix = np.stack([
        np.stack([cos_y, -sin_y], axis=-1),
        np.stack([sin_y, cos_y], axis=-1)
    ], axis=-2)

    # 使用 matmul (@) 进行批次矩阵乘法，它比 einsum 更直观
    # offsets: (..., 4, 2) -> (..., 4, 1, 2)
    # rot_matrix: (..., 2, 2) -> (..., 1, 2, 2)
    # result: (..., 4, 1, 2)
    rotated_offsets = offsets[..., np.newaxis, :] @ rot_matrix[..., np.newaxis, :, :]
    rotated_offsets = np.squeeze(rotated_offsets, axis=-2)

    # 添加到中心点
    # pos: [..., 2] -> [..., 1, 2]
    corners = pos[..., np.newaxis, :] + rotated_offsets

    return corners


def calculate_map_violation_penalty(
        trajectory: np.ndarray,
        map_features: dict,
        traffic_motion_feat: dict,
        adv_info: dict,
        cross_solid_line_scale=50.0,
        crash_object_scale=10.0
) -> dict:
    """
    计算违规惩罚 (V3 - 高效优化版)。
    - 使用向量化和宽相/窄相检测策略优化与其他车辆的碰撞检测。
    """
    penalties = {
        "cross_solid_line_penalty": 0.0,
        "crash_object_penalty": 0.0,
    }

    # --- 1. 穿越地图边界线检测 (这部分已经很快，无需大改) ---
    impassable_walls = []
    impassable_line_types = {
        'ROAD_EDGE_BOUNDARY',
        # 'ROAD_EDGE_MEDIAN'
    }

    for feature_id, feature_data in map_features.items():
        if feature_data.get('type') in impassable_line_types and 'polyline' in feature_data:
            polyline = feature_data['polyline'][:, :2]
            if len(polyline) >= 2:
                from shapely.geometry import LineString
                impassable_walls.append(LineString(polyline))
    wall_tree = STRtree(impassable_walls) if impassable_walls else None

    yaw_adv_traj = get_polyline_yaw(trajectory[:, :2])
    width_adv, length_adv = np.array(adv_info['w']), np.array(adv_info['l'])

    if wall_tree and len(trajectory) > 1:
        adv_pos = trajectory[:, :2]
        # 使用向量化函数一次性计算所有时刻的角点
        adv_all_corners = _vectorized_get_corners(adv_pos, yaw_adv_traj, length_adv, width_adv)
        violation_points = 0
        for t in range(len(trajectory)):
            adv_polygon = Polygon(adv_all_corners[t])
            if any(adv_polygon.intersects(wall) for wall in wall_tree.query(adv_polygon)):
                violation_points += 1
        if violation_points > 0:
            penalties["cross_solid_line_penalty"] = cross_solid_line_scale
    # --- 地图检测结束 ---

    # --- 2. 与其他车辆碰撞检测 (高效优化版) ---
    num_vehicles = traffic_motion_feat['state/future/x'].shape[0]
    # --- 向量化数据准备 ---
    all_x = traffic_motion_feat['state/future/x'].numpy()
    all_y = traffic_motion_feat['state/future/y'].numpy()
    all_pos = np.stack([all_x, all_y], axis=-1)  # Shape: [num_vehicles, num_timesteps, 2]
    all_yaw = traffic_motion_feat['state/future/bbox_yaw'].numpy()
    all_len = traffic_motion_feat['state/future/length'].numpy()
    all_wid = traffic_motion_feat['state/future/width'].numpy()
    all_valid = traffic_motion_feat['state/future/valid'].numpy()

    num_timesteps = len(trajectory)

    # --- 2a. 宽相检测 (Broad Phase) ---

    # 向量化计算所有车辆在所有时刻的角点
    # valid_mask: [num_vehicles, num_timesteps, 1, 1]
    valid_mask = all_valid[..., np.newaxis, np.newaxis] > 0
    all_corners = _vectorized_get_corners(all_pos, all_yaw, all_len, all_wid)

    # 计算每个车辆整个轨迹的AABB (min_x, min_y, max_x, max_y)
    # 使用valid_mask确保只考虑有效点, 无效点设为inf/-inf以不影响min/max
    all_corners_masked = np.where(valid_mask, all_corners, np.inf)
    min_coords = np.min(all_corners_masked, axis=(1, 2))  # Shape: [num_vehicles, 2]
    all_corners_masked = np.where(valid_mask, all_corners, -np.inf)
    max_coords = np.max(all_corners_masked, axis=(1, 2))  # Shape: [num_vehicles, 2]

    # 对抗车的轨迹AABB
    adv_traj_corners = _vectorized_get_corners(trajectory[:, :2], yaw_adv_traj, length_adv, width_adv)
    adv_min_coord = np.min(adv_traj_corners, axis=(0, 1))
    adv_max_coord = np.max(adv_traj_corners, axis=(0, 1))
    adv_trajectory_box = box(adv_min_coord[0], adv_min_coord[1], adv_max_coord[0], adv_max_coord[1])

    # 筛选出可能与对抗车轨迹AABB碰撞的其他车辆
    potential_collision_indices = []
    # 从索引2开始，跳过主车和对抗车
    for i in range(2, num_vehicles):
        # 如果车辆没有任何有效时刻，则跳过
        if np.all(min_coords[i] == np.inf): continue

        other_vehicle_traj_box = box(min_coords[i, 0], min_coords[i, 1], max_coords[i, 0], max_coords[i, 1])
        if adv_trajectory_box.intersects(other_vehicle_traj_box):
            potential_collision_indices.append(i)

    if not potential_collision_indices:
        return penalties  # 宽相检测未发现任何潜在碰撞，提前返回

    # --- 2b. 窄相检测 (Narrow Phase) ---
    # 只对少数潜在碰撞者进行精确检测
    # 提取潜在碰撞车辆的角点数据
    candidate_corners = all_corners[potential_collision_indices]  # Shape: [num_candidates, T, 4, 2]

    for t in range(num_timesteps):
        adv_polygon = Polygon(adv_traj_corners[t])

        for i, original_idx in enumerate(potential_collision_indices):
            # 检查该车在该时刻是否有效
            if all_valid[original_idx, t]:
                other_polygon = Polygon(candidate_corners[i, t])
                if adv_polygon.intersects(other_polygon):
                    penalties["crash_object_penalty"] = crash_object_scale
                    return penalties  # 找到第一个碰撞就立即返回

    return penalties


# ==========================================================================================
def calculate_realism_penalty(trajectory: np.ndarray, adv_info: dict) -> dict:
    """
    计算轨迹的真实性惩罚 (V3: 平衡尺度，平滑惩罚)
    - 惩罚运动学极限 (高阶加速度/角速度)
    - 惩罚不合理的行为模式 (长时间静止，大角度转向)
    """
    if len(trajectory) < 5:
        return {
            "kinematic_penalty": 0.0,
            "behavior_penalty": 0.0
        }

    dt = 0.1
    positions = trajectory[:, :2]

    # --- 基础运动学量计算 ---
    velocities = np.gradient(positions, dt, axis=0, edge_order=2)
    speeds = np.linalg.norm(velocities, axis=1)
    # accelerations = np.gradient(velocities, dt, axis=0, edge_order=2)

    headings = get_polyline_yaw(positions)
    unwrapped_headings = np.unwrap(headings)
    angular_velocities = np.gradient(unwrapped_headings, dt, edge_order=2)

    # --- 1. 运动学极限惩罚 (可行性) ---
    # 合理的加速度范围约在 [-5, 5] m/s^2, 角速度在 [-1, 1] rad/s
    # 我们希望惩罚超出这些范围的值
    longitudinal_accel = np.gradient(speeds, dt, edge_order=2)
    lateral_accel = speeds * angular_velocities

    # 使用 softplus 来平滑地惩罚超出舒适区的值
    # softplus(x) = log(1 + exp(x))
    # 惩罚正向/负向加速度过大的情况
    accel_comfort_zone = 7.0  # m/s^2
    lat_accel_comfort_zone = 6.0  # m/s^2
    ang_vel_comfort_zone = 0.8  # rad/s

    accel_penalty = np.mean(np.log1p(np.exp(np.abs(longitudinal_accel) - accel_comfort_zone)))
    lat_accel_penalty = np.mean(np.log1p(np.exp(np.abs(lateral_accel) - lat_accel_comfort_zone)))
    ang_vel_penalty = np.mean(np.log1p(np.exp(np.abs(angular_velocities) - ang_vel_comfort_zone)))

    kinematic_penalty_factor_accel = 5.0
    kinematic_penalty_factor_ang_vel = 5.0

    kinematic_penalty = kinematic_penalty_factor_accel * (accel_penalty + lat_accel_penalty) + \
                        kinematic_penalty_factor_ang_vel * ang_vel_penalty

    # --- 2. 不合理行为模式惩罚 (平滑化处理) ---
    # 移除硬阈值 "悬崖", 改为平滑的惩罚函数

    # b. 惩罚过大的总航向变化
    # 总转向不应超过一个大弯(e.g., 180度), 惩罚超出部分
    high_turn_penalty_factor = 5.0
    total_heading_change = np.abs(unwrapped_headings[-1] - unwrapped_headings[0])
    max_reasonable_turn = np.pi  # 180 degrees
    high_turn_penalty = high_turn_penalty_factor * np.log1p(np.exp(total_heading_change - max_reasonable_turn))

    # c. 惩罚低速下的剧烈转向 (原地打转)
    # 惩罚 "角速度" 和 "速度的倒数" 的乘积
    stop_and_turn_penalty_factor = 3.0
    # 加一个很小的数避免除以零
    turn_while_slow_metric = np.abs(angular_velocities) / (speeds + 1e-2)
    stop_and_turn_penalty = stop_and_turn_penalty_factor * np.mean(turn_while_slow_metric)

    # 将所有行为惩罚合并
    behavior_penalty = high_turn_penalty + stop_and_turn_penalty

    return {
        "kinematic_penalty": kinematic_penalty,
        "behavior_penalty": behavior_penalty,
    }


def calculate_adversarial_reward(
        adv_traj: np.ndarray, ego_traj: np.ndarray, adv_info: dict, ego_info: dict
) -> tuple[float, bool]:
    """
    计算对抗性奖励。结合了碰撞奖励和基于距离的接近奖励。
    返回 (总奖励, 是否碰撞) 的元组，方便统计碰撞率。
    """
    # ------------------- 修改点 1: 移除下采样 -------------------
    # 直接使用原始轨迹，但仍需对齐长度
    common_len = min(len(adv_traj), len(ego_traj))
    if common_len < 2:
        return 0.0, False

    # 根据公共长度截断轨迹数据
    adv_traj_aligned = adv_traj[:common_len]
    ego_traj_aligned = ego_traj[:common_len]
    # ------------------- 修改结束 -------------------

    # --- 碰撞检测 (现在在更高频率的轨迹上进行) ---
    is_collision = False
    collision_time_step = -1  # 用于记录碰撞发生的时间

    # 注意：这里的yaw和bbox计算现在会处理更多的点
    yaw_adv = get_polyline_yaw(adv_traj_aligned).reshape(-1, 1)
    width_adv, length_adv = adv_info['w'], adv_info['l']
    yaw_ego = get_polyline_yaw(ego_traj_aligned).reshape(-1, 1)
    width_ego, length_ego = ego_info['w'], ego_info['l']

    # Bbox函数
    def get_bbox(traj, yaw, w, l):
        cos_theta, sin_theta = np.cos(yaw), np.sin(yaw)
        p1 = traj + np.hstack([0.5 * l * cos_theta + 0.5 * w * sin_theta, 0.5 * l * sin_theta - 0.5 * w * cos_theta])
        p2 = traj + np.hstack([0.5 * l * cos_theta - 0.5 * w * sin_theta, 0.5 * l * sin_theta + 0.5 * w * cos_theta])
        p3 = traj + np.hstack([-0.5 * l * cos_theta - 0.5 * w * sin_theta, -0.5 * l * sin_theta + 0.5 * w * cos_theta])
        p4 = traj + np.hstack([-0.5 * l * cos_theta + 0.5 * w * sin_theta, -0.5 * l * sin_theta - 0.5 * w * cos_theta])
        return p1, p2, p3, p4

    adv_p1, adv_p2, adv_p3, adv_p4 = get_bbox(adv_traj_aligned, yaw_adv, width_adv, length_adv)
    ego_p1, ego_p2, ego_p3, ego_p4 = get_bbox(ego_traj_aligned, yaw_ego, width_ego, length_ego)

    # 这里的循环现在是安全的
    for t in range(common_len):
        adv_corners = [adv_p1[t], adv_p2[t], adv_p3[t], adv_p4[t]]
        ego_corners = [ego_p1[t], ego_p2[t], ego_p3[t], ego_p4[t]]
        adv_edges = [(adv_corners[i], adv_corners[(i + 1) % 4]) for i in range(4)]
        ego_edges = [(ego_corners[i], ego_corners[(i + 1) % 4]) for i in range(4)]

        collision_found_this_step = False
        for adv_e in adv_edges:
            for ego_e in ego_edges:
                l1 = np.concatenate(adv_e)
                l2 = np.concatenate(ego_e)
                if Intersect(l1, l2):
                    is_collision = True
                    collision_time_step = t
                    collision_found_this_step = True
                    break  # 退出 ego_e 循环
            if collision_found_this_step:
                break  # 退出 adv_e 循环

        if is_collision:
            break  # 找到第一个碰撞时刻就退出主循环 t

    # --- 奖励计算 ---
    # 如果发生碰撞，奖励主要由碰撞奖励构成
    collision_reward_scale = 10.0
    if is_collision:
        # 奖励与碰撞发生的时间成反比，越早碰撞奖励越高
        collision_rew = collision_reward_scale * (1.0 - collision_time_step / common_len)
        return collision_rew, True

    # 如果没有碰撞，则计算接近奖励和引导奖励
    # ------------------- 修改点 2: 在这里也使用对齐后的轨迹 -------------------
    distances = np.linalg.norm(adv_traj_aligned - ego_traj_aligned, axis=1)
    min_dist = np.min(distances)
    proximity_reward = 0.0
    proximity_reward_scale = 1.0
    proximity_decay_rate = 0.2
    if min_dist <= 20:
        proximity_reward = proximity_reward_scale * np.exp(-proximity_decay_rate * min_dist)

    return proximity_reward, False


def dpo_loss(policy_log_probs_w: torch.Tensor, policy_log_probs_l: torch.Tensor,
             ref_log_probs_w: torch.Tensor, ref_log_probs_l: torch.Tensor,
             beta: float) -> torch.Tensor:
    log_ratio_policy = policy_log_probs_w - policy_log_probs_l
    with torch.no_grad():
        log_ratio_ref = ref_log_probs_w - ref_log_probs_l
    loss = -F.logsigmoid(beta * (log_ratio_policy - log_ratio_ref))
    return loss


from sage.rewards import (  # noqa: E402
    calculate_adversarial_reward,
    calculate_map_violation_penalty,
    calculate_realism_penalty,
)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a motion forecasting model with Group-Sampled DPO")
    # DPO specific arguments
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="Learning rate for DPO fine-tuning")
    parser.add_argument('--beta', type=float, default=0.05, help="Beta parameter for DPO loss")
    parser.add_argument('--epochs', type=int, default=200, help="Number of training epochs")


    parser.add_argument('--num_pairs_per_group', type=int, default=8,  # 8 is the best
                        help="Number of preference pairs to sample from each group of candidate trajectories.")
    parser.add_argument('--reward_margin', type=float, default=0.2,
                        help="Minimum reward difference to consider a pair valid. Prevents learning from noisy pairs.")

    parser.add_argument('--adversarial_weight', type=float, default=1.0, help="Weight for the adversarial penalty")
    parser.add_argument('--realism_weight', type=float, default=10.0, help="Weight for the realism penalty")


    parser.add_argument('--save_path', type=str, default='./advgen/finetuned/hgpo_finetuned_model_real.bin',
                        help="Path to save the fine-tuned model")
    parser.add_argument('--base_model_path', type=str, default='./advgen/pretrained/densetnt.bin',
                        help="Path to the pretrained DenseTNT checkpoint")
    parser.add_argument('--data_directory', type=str, default='./raw_scenes_500',
                        help="Path to processed MetaDrive/WOMD scenarios")
    parser.add_argument('--split_file', type=str, default='./configs/splits/sage_womd_500.json',
                        help="Scenario split JSON file")
    parser.add_argument('--split', type=str, default='train', help="Split name in the split JSON")
    parser.add_argument('--max_scenarios', type=int, default=None,
                        help="Optional cap for smoke tests")
    parser.add_argument('--scenario_csv_path', type=str, default='./configs/splits/sage_autopilot_summary.csv',
                        help="Path to the scenario summary CSV")
    parser.add_argument('--log_dir', type=str, default='runs', help="Base directory for TensorBoard logs")
    parser.add_argument('--run_name', type=str, default='hgpo_finetune',
                        help="A descriptive name for the current run")

    dpo_args = parser.parse_args()
    print("Group-Sampled DPO Training Arguments:", dpo_args)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_log_dir = os.path.join(dpo_args.log_dir, f"{dpo_args.run_name}_{timestamp}")
    writer = SummaryWriter(log_dir=run_log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    train_ids = scenario_ids(dpo_args.split_file, split=dpo_args.split, max_scenarios=dpo_args.max_scenarios)
    train_ids = filter_ids_by_summary(train_ids, dpo_args.scenario_csv_path)
    print(f"Found {len(train_ids)} scenarios to train on after filtering.")
    if not train_ids:
        print("Error: No scenarios left after filtering. Please check the split and scenario summary files.")
        return

    adv_gen_parser = argparse.ArgumentParser()
    dpo_adv_generator = DPOAdvGenerator(adv_gen_parser, model_path=dpo_args.base_model_path)

    env = WaymoEnv({
        "agent_policy": ReplayEgoCarPolicy,
        "reactive_traffic": False,
        "use_render": False,
        "data_directory": dpo_args.data_directory,
        "start_scenario_index": 0,
        "num_scenarios": max(train_ids) + 1,
        "force_reuse_object_name": True,
        "sequential_seed": True,
    })

    optimizer = torch.optim.AdamW(dpo_adv_generator.policy_model.parameters(), lr=dpo_args.learning_rate)
    import random
    all_train_ids = list(train_ids)

    global_step = 0
    best_avg_winner_reward = -float('inf')
    best_model_path = dpo_args.save_path.replace('.bin', '_best.bin')

    for epoch in range(dpo_args.epochs):
        print(f"\n--- Starting Epoch {epoch + 1}/{dpo_args.epochs} ---")
        random.shuffle(all_train_ids)

        epoch_metrics = {
            "total_loss": 0.0, "scenarios_processed": 0, "scenarios_skipped": 0, "pairs_processed": 0,
            "avg_winner_reward": [], "avg_loser_reward": [],
            "avg_winner_adv_rew": [], "collision_rate": [],
            "avg_winner_real_pen": [], "avg_winner_map_pen": [],
            "avg_winner_pen_behavior": [], "avg_winner_pen_kinematic": [],
            "avg_winner_pen_crash_object": [], "avg_winner_pen_cross_solid_line": [],
            "avg_feasibility_rate_all_candidates": [],
        }

        pbar = tqdm(all_train_ids, desc=f"Epoch {epoch + 1}")
        for seed in pbar:
            try:
                env.reset(force_seed=seed)
            except Exception as e:
                print(f"Warning: Failed to reset environment for seed {seed}. Error: {e}. Skipping.")
                epoch_metrics["scenarios_skipped"] += 1
                continue

            batch_data, traffic_motion_feat, adv_info, ego_info, ego_gt_future_traj, adv_past_traj, raw_map_features = dpo_adv_generator.get_data_for_scenario(
                env)
            pred_trajs_list_np, pred_scores_list_t, _ = dpo_adv_generator.policy_model(
                batch_data[0], device, return_tensors_for_dpo=True
            )
            adv_candidate_trajs_np = pred_trajs_list_np[1]
            adv_candidate_log_probs = pred_scores_list_t[1]

            # ======================= C-DPO 核心修改 =======================

            trajectory_info = []
            for i, traj in enumerate(adv_candidate_trajs_np):
                # 1. 计算所有奖励和惩罚分量
                map_violations = calculate_map_violation_penalty(traj, raw_map_features, traffic_motion_feat, adv_info)
                total_map_penalty = map_violations["cross_solid_line_penalty"] + map_violations["crash_object_penalty"]

                realism_penalties = calculate_realism_penalty(traj, adv_info)
                total_realism_penalty = realism_penalties["behavior_penalty"] + realism_penalties["kinematic_penalty"]

                adversarial_rew, is_collision = calculate_adversarial_reward(traj, ego_gt_future_traj, adv_info,
                                                                             ego_info)

                # 2. 计算用于偏好排序的奖励 (不再包含地图惩罚)
                preference_reward = (dpo_args.adversarial_weight * adversarial_rew -
                                     dpo_args.realism_weight * total_realism_penalty)

                # 3. 判断可行性
                is_feasible = (total_map_penalty == 0)

                trajectory_info.append({
                    "index": i,
                    "is_feasible": is_feasible,
                    "preference_reward": preference_reward,
                    # 存储其他信息用于日志记录
                    "total_reward_for_log": preference_reward - total_map_penalty,  # 用于和旧方法比较
                    "adv_rew": adversarial_rew,
                    "is_collision": is_collision,
                    "real_pen_total": total_realism_penalty,
                    "map_pen_total": total_map_penalty,
                    **realism_penalties, **map_violations
                })

            # 4. 将轨迹分为可行集和不可行集
            feasible_trajs = [info for info in trajectory_info if info["is_feasible"]]
            infeasible_trajs = [info for info in trajectory_info if not info["is_feasible"]]

            # 计算并记录当前场景所有候选轨迹的真实可行性率
            num_total_candidates = len(trajectory_info)
            num_feasible_candidates = len(feasible_trajs)
            avg_feasibility_this_scenario = num_feasible_candidates / num_total_candidates if num_total_candidates > 0 else 0.0
            epoch_metrics["avg_feasibility_rate_all_candidates"].append(avg_feasibility_this_scenario)


            if len(feasible_trajs) < 1 or len(feasible_trajs) + len(infeasible_trajs) < 2:
                epoch_metrics["scenarios_skipped"] += 1
                continue

            with torch.no_grad():
                _, ref_scores_list, _ = dpo_adv_generator.reference_model(batch_data[0], device,
                                                                          return_tensors_for_dpo=True)
                ref_log_probs = ref_scores_list[1]

            scenario_total_loss = 0.0
            pairs_found = 0

            # 5. 根据C-DPO规则采样偏好对
            preference_pairs = []

            # 如果没有可行轨迹，或者只有一个轨迹，无法形成偏好对，则跳过
            if not feasible_trajs or len(trajectory_info) < 2:
                epoch_metrics["scenarios_skipped"] += 1
                continue

            # 规则 1 (强化版): 每个不可行轨迹都必须是 loser
            # 对于每一个不可行的轨迹，都随机从可行集中选择一个 winner 与之配对。
            if infeasible_trajs:
                for loser_info in infeasible_trajs:
                    winner_info = random.choice(feasible_trajs)
                    preference_pairs.append((winner_info, loser_info))

            # 规则 2: 在可行集中学习权衡 (Adversarial vs. Realism)
            # 使用剩余的采样预算在可行集中寻找偏好对。
            remaining_budget = max(0, dpo_args.num_pairs_per_group - len(preference_pairs))
            if len(feasible_trajs) >= 2 and remaining_budget > 0:
                for _ in range(remaining_budget):
                    winner_candidate, loser_candidate = random.sample(feasible_trajs, 2)

                    # 确保是正确的winner/loser对
                    if winner_candidate["preference_reward"] > loser_candidate["preference_reward"]:
                        winner, loser = winner_candidate, loser_candidate
                    else:
                        winner, loser = loser_candidate, winner_candidate

                    # 检查奖励差异是否足够大
                    if winner["preference_reward"] - loser["preference_reward"] > dpo_args.reward_margin:
                        preference_pairs.append((winner, loser))

            if not preference_pairs:
                epoch_metrics["scenarios_skipped"] += 1
                continue

            # 6. 计算loss并优化
            for winner_info, loser_info in preference_pairs:
                winner_idx = winner_info["index"]
                loser_idx = loser_info["index"]

                policy_log_probs_w = adv_candidate_log_probs[winner_idx]
                policy_log_probs_l = adv_candidate_log_probs[loser_idx]
                ref_log_probs_w = ref_log_probs[winner_idx]
                ref_log_probs_l = ref_log_probs[loser_idx]

                pair_loss = dpo_loss(
                    policy_log_probs_w, policy_log_probs_l,
                    ref_log_probs_w.detach(), ref_log_probs_l.detach(),
                    beta=dpo_args.beta
                )
                scenario_total_loss += pair_loss
                pairs_found += 1

            if pairs_found == 0:
                epoch_metrics["scenarios_skipped"] += 1
                continue

            average_scenario_loss = scenario_total_loss / pairs_found

            optimizer.zero_grad()
            average_scenario_loss.backward()
            optimizer.step()
            # ======================= C-DPO 核心修改结束 =======================

            # --- 日志记录与评估 ---
            # 为了日志记录，我们需要确定一个“代表性”的最佳轨迹。
            # 规则：
            # 1. 如果存在可行轨迹，则“最佳”轨迹是可行集中偏好奖励最高的那个。
            # 2. 如果没有可行轨迹，则“最佳”轨迹是所有轨迹中（包括不可行的）总奖励最高的那个，以便我们观察到模型“最不差”的尝试。
            if feasible_trajs:
                best_traj_for_log = max(feasible_trajs, key=lambda x: x["preference_reward"])
            else:
                best_traj_for_log = max(trajectory_info, key=lambda x: x["total_reward_for_log"])

            # 为了比较，我们记录一个“最差”轨迹
            worst_traj_for_log = min(trajectory_info, key=lambda x: x["total_reward_for_log"])

            # 更新 Epoch 级别的统计数据
            epoch_metrics["total_loss"] += average_scenario_loss.item()
            epoch_metrics["scenarios_processed"] += 1
            epoch_metrics["pairs_processed"] += pairs_found

            # 使用我们为日志选择的“最佳”轨迹来填充指标
            epoch_metrics["avg_winner_reward"].append(best_traj_for_log['total_reward_for_log'])
            epoch_metrics["avg_loser_reward"].append(worst_traj_for_log['total_reward_for_log'])
            epoch_metrics["avg_winner_adv_rew"].append(best_traj_for_log['adv_rew'])
            epoch_metrics["collision_rate"].append(1.0 if best_traj_for_log['is_collision'] else 0.0)
            epoch_metrics["avg_winner_real_pen"].append(best_traj_for_log['real_pen_total'])
            epoch_metrics["avg_winner_map_pen"].append(best_traj_for_log['map_pen_total'])
            epoch_metrics["avg_winner_pen_behavior"].append(best_traj_for_log['behavior_penalty'])
            epoch_metrics["avg_winner_pen_kinematic"].append(best_traj_for_log['kinematic_penalty'])
            epoch_metrics["avg_winner_pen_crash_object"].append(best_traj_for_log['crash_object_penalty'])
            epoch_metrics["avg_winner_pen_cross_solid_line"].append(best_traj_for_log['cross_solid_line_penalty'])

            # 更新 TensorBoard (per step)
            writer.add_scalar('Step/Loss', average_scenario_loss.item(), global_step)
            writer.add_scalar('Step/Reward/Best_Traj_Total_Reward', best_traj_for_log['total_reward_for_log'],
                              global_step)
            writer.add_scalar('Step/Reward/Best_Traj_Preference_Reward', best_traj_for_log['preference_reward'],
                              global_step)
            writer.add_scalar('Step/Feasibility/Is_Best_Traj_Feasible', float(best_traj_for_log['is_feasible']),
                              global_step)
            writer.add_scalar('Step/Feasibility/Num_Feasible_Candidates', len(feasible_trajs), global_step)

            global_step += 1

            # 更新 tqdm 进度条
            pbar.set_postfix({
                "loss": f"{average_scenario_loss.item():.3f}",
                "best_rew": f"{best_traj_for_log['total_reward_for_log']:.2f}",
                "feasible": f"{len(feasible_trajs)}/{len(trajectory_info)}",
                "pairs": f"{pairs_found}",
                "skip": epoch_metrics["scenarios_skipped"]
            })

            # --- Epoch 结束，计算并打印总结 ---
        if epoch_metrics["scenarios_processed"] > 0:
            avg_loss = epoch_metrics["total_loss"] / epoch_metrics["scenarios_processed"]
            avg_win_rew = np.mean(epoch_metrics["avg_winner_reward"])
            avg_lose_rew = np.mean(epoch_metrics["avg_loser_reward"])
            avg_win_adv_rew = np.mean(epoch_metrics["avg_winner_adv_rew"])
            avg_win_real_pen = np.mean(epoch_metrics["avg_winner_real_pen"])
            avg_win_map_pen = np.mean(epoch_metrics["avg_winner_map_pen"])
            avg_coll_rate = np.mean(epoch_metrics["collision_rate"])
            avg_pairs_per_scenario = epoch_metrics["pairs_processed"] / epoch_metrics["scenarios_processed"]

            # 计算最佳轨迹的可行率
            # 注意：这里的avg_win_map_pen是惩罚值，所以可行率是惩罚为0的比例
            avg_feasibility_rate = np.mean(np.array(epoch_metrics["avg_winner_map_pen"]) == 0)  # 仅最佳轨迹
            avg_feasibility_all_candidates = np.mean(epoch_metrics["avg_feasibility_rate_all_candidates"])

            # 检查是否是最佳模型并保存
            # 评价标准：优先考虑可行性，其次是总奖励。一个简单的复合指标可以是: `avg_win_rew * (1 + avg_feasibility_rate)`
            # 或者更直接地，我们仍然可以用 avg_win_rew，因为高地图惩罚会拉低这个值。
            if avg_win_rew > best_avg_winner_reward:
                best_avg_winner_reward = avg_win_rew
                print(
                    f"\n[Model Save] New best model found at epoch {epoch + 1} with Avg Winner Reward: {avg_win_rew:.4f}")
                print(f"  -> Corresponding Feasibility Rate: {avg_feasibility_rate:.2%}")
                print(f"  -> Saving model to {best_model_path}")
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                torch.save(dpo_adv_generator.policy_model.state_dict(), best_model_path)

            print(f"\n--- Epoch {epoch + 1} Summary ---")
            print(f"  Avg Loss: {avg_loss:.4f} | Avg Pairs per Scenario: {avg_pairs_per_scenario:.2f}")
            print(
                f"  Avg 'Best Traj' Total Reward: {avg_win_rew:.2f} | Avg 'Worst Traj' Total Reward: {avg_lose_rew:.2f}")
            print(
                f"  'Best Traj' Breakdown -> Adv Rew: {avg_win_adv_rew:.2f}, Real Pen: {avg_win_real_pen:.2f}, Map Pen: {avg_win_map_pen:.2f}")
            print(
                f"  'Best Traj' Collision Rate: {avg_coll_rate:.2%} | 'Best Traj' Feasibility Rate: {avg_feasibility_rate:.2%}")
            print(f"  Avg Feasibility Rate (All Candidates - The Truth): {avg_feasibility_all_candidates:.2%}")
            print(
                f"  Scenarios Processed: {epoch_metrics['scenarios_processed']} | Skipped: {epoch_metrics['scenarios_skipped']}")

            # 记录 Epoch 级别的 TensorBoard 日志
            writer.add_scalar('Epoch/Loss', avg_loss, epoch)
            writer.add_scalar('Epoch/Metrics/Feasibility_Rate', avg_feasibility_rate, epoch)
            writer.add_scalar('Epoch/Metrics/Feasibility_Rate_AllCandidates', avg_feasibility_all_candidates, epoch)
            writer.add_scalar('Epoch/Metrics/Collision_Rate', avg_coll_rate, epoch)
            writer.add_scalar('Epoch/Metrics/Avg_Best_Traj_Total_Reward', avg_win_rew, epoch)
            writer.add_scalar('Epoch/Metrics/Avg_Pairs_per_Scenario', avg_pairs_per_scenario, epoch)

            # 记录最佳轨迹奖励的组成成分
            writer.add_scalar('Epoch/Avg_BestTraj_Components/1_Adversarial_Reward', avg_win_adv_rew, epoch)
            writer.add_scalar('Epoch/Avg_BestTraj_Components/2_Realism_Penalty', avg_win_real_pen, epoch)
            writer.add_scalar('Epoch/Avg_BestTraj_Components/3_Map_Penalty', avg_win_map_pen, epoch)

            writer.add_scalar('Epoch/Avg_Penalties/Behavior', np.mean(epoch_metrics['avg_winner_pen_behavior']), epoch)
            writer.add_scalar('Epoch/Avg_Penalties/Kinematic', np.mean(epoch_metrics['avg_winner_pen_kinematic']),
                              epoch)
            writer.add_scalar('Epoch/Avg_Penalties/CrashObject', np.mean(epoch_metrics['avg_winner_pen_crash_object']),
                              epoch)
            writer.add_scalar('Epoch/Avg_Penalties/CrossSolidLine',
                              np.mean(epoch_metrics['avg_winner_pen_cross_solid_line']), epoch)


    print(f"\nTraining finished. Saving final model to {dpo_args.save_path}")
    os.makedirs(os.path.dirname(dpo_args.save_path), exist_ok=True)
    torch.save(dpo_adv_generator.policy_model.state_dict(), dpo_args.save_path)
    print(f"Final model saved. The best performing model was saved to {best_model_path}")

    writer.close()
    env.close()


if __name__ == '__main__':
    main()
