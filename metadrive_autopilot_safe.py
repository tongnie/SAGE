# metadrive_autopilot.py

import math
import heapq
import numpy as np
from metadrive.policy.base_policy import BasePolicy
from metadrive.component.vehicle_module.PID_controller import PIDController

from metadrive.engine.engine_utils import get_engine
from metadrive.policy.base_policy import BasePolicy
from metadrive.utils.math import wrap_to_pi, norm
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from copy import deepcopy
from collections import deque
import matplotlib.pyplot as plt

class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=40):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._saved_window = deque([0 for _ in range(n)], maxlen=n)
        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        if len(self._window) >= 2:
            integral = sum(self._window)/len(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0
        return self._K_P * error + self._K_I * integral + self._K_D * derivative

    def save(self):
        self._saved_window = deepcopy(self._window)

    def load(self):
        self._window = self._saved_window

    def reset(self):
        self.error = 0
        self.integral = 0
        self.derivative = 0

class KinematicBicycleModel:
    """
    运动学自行车模型，从CARLA Autopilot代码[110]迁移。
    用于预测车辆未来的位置、朝向和速度。
    """
    def __init__(self, dt=0.1): # 论文描述的预测步长 50ms
        self.dt = dt
        # 这些参数来自论文[110]或CARLA Autopilot的常见实现
        self.front_wb = 1.2 # 近似值, L_f (前轴到重心的距离)
        self.rear_wb = 1.3  # 近似值, L_r (后轴到重心的距离)
        # ^^^ 注意: 这些轴距参数对模型精度影响很大，最好能从MetaDrive车辆配置获取
        # 或使用CARLA Autopilot中的精确值
        # CARLA Autopilot中的值:
        # self.front_wb = -0.090769015 # 这似乎是相对于重心的偏移, 而不是轴距
        # self.rear_wb = 1.4178275
        # 这里的 L = front_wb + rear_wb 是车辆的轴距 (wheelbase)
        # 我们用更直观的 L_f, L_r, 假设重心在中间附近
        
        # 如果MetaDrive车辆有wheelbase属性:
        # vehicle_length = 4.5 # 假设
        # self.wheelbase = vehicle_length * 0.6 # 估算轴距
        # self.L_f = self.wheelbase * 0.5
        # self.L_r = self.wheelbase * 0.5
        # 更精确的:
        # L = 2.5 # 假设车辆轴距
        # self.L_f = L / 2
        # self.L_r = L / 2
        # 论文[110] (World on Rails) 中的车辆参数可能更合适
        # 例如 L = 2.65m, lr = 1.325m (后轴到重心), lf = L - lr
        # 我们这里暂时使用CARLA Autopilot中的参数来保持一致性，但要理解其含义
        # 这些参数实际上是用于计算侧滑角 beta 的，而不是直接的 lf, lr
        self.CARLA_WB_PARAM_REAR = 1.4178275 # 对应 beta 公式中的 L_r
        self.CARLA_WB_PARAM_TOTAL_INV = 1.0 / (self.CARLA_WB_PARAM_REAR - 0.090769015) # 对应 1/(L_f+L_r)

        self.steer_gain = 0.36848336    # 转向控制到实际舵角的增益
        self.brake_accel = -4.952399   # 刹车时的最大减速度 (m/s^2)
        self.throt_accel = 0.5633837   # 油门为1时的加速度系数 (m/(s^2 * throttle_unit))

    def forward(self, pos, heading_rad, speed_m_s, control_action):
        """
        向前预测一步。
        pos: [x, y] 当前位置 (m)
        heading_rad: 当前朝向 (弧度, 标准数学坐标系, 0朝东, 逆时针)
        speed_m_s: 当前速度 (m/s)
        control_action: [steer, throttle, brake]
                        steer: [-1, 1]
                        throttle: [0, 1]
                        brake: 0 or 1
        """
        steer_norm, throttle_norm, brake_flag = control_action
        
        # 将归一化的转向 [-1,1] 转换为实际舵角 (弧度)
        wheel_angle_rad = self.steer_gain * steer_norm
        
        if brake_flag > 0.5: # 假设刹车是 0 或 1
            acceleration = self.brake_accel
        else:
            acceleration = self.throt_accel * throttle_norm
            
        # 运动学自行车模型公式
        # beta = math.atan( (self.L_r / (self.L_f + self.L_r)) * math.tan(wheel_angle_rad) )
        # 使用CARLA参数的beta计算
        beta = math.atan( self.CARLA_WB_PARAM_REAR * self.CARLA_WB_PARAM_TOTAL_INV * math.tan(wheel_angle_rad) )

        next_pos_x = pos[0] + speed_m_s * math.cos(heading_rad + beta) * self.dt
        next_pos_y = pos[1] + speed_m_s * math.sin(heading_rad + beta) * self.dt
        
        # next_heading_rad = heading_rad + (speed_m_s / self.L_r) * math.sin(beta) * self.dt
        # 使用CARLA参数的朝向更新 (后轴参考点)
        next_heading_rad = heading_rad + (speed_m_s / self.CARLA_WB_PARAM_REAR) * math.sin(beta) * self.dt
        next_heading_rad = wrap_to_pi(next_heading_rad) # 保持在 [-pi, pi]

        next_speed_m_s = speed_m_s + acceleration * self.dt
        next_speed_m_s = max(0.0, next_speed_m_s) # 速度不能为负

        return np.array([next_pos_x, next_pos_y]), next_heading_rad, next_speed_m_s

# OBB (Oriented Bounding Box) 定义
class OBB:
    def __init__(self, center_xy, width, length, heading_rad):
        self.center = np.array(center_xy)
        self.width = width   # 垂直于朝向的长度
        self.length = length # 平行于朝向的长度
        self.heading_rad = heading_rad # 标准数学坐标系角度

        # 计算半长和半宽，方便后续计算
        self.half_width = self.width / 2.0
        self.half_length = self.length / 2.0

        # 计算OBB的四个角点 (相对于OBB中心, 在OBB局部坐标系下)
        # local_corners: (front_left, front_right, rear_right, rear_left)
        self._local_corners = np.array([
            [-self.half_length,  self.half_width],
            [ self.half_length,  self.half_width],
            [ self.half_length, -self.half_width],
            [-self.half_length, -self.half_width]
        ])
        
        # 计算旋转到世界坐标系的角点
        R = np.array([
            [math.cos(self.heading_rad), -math.sin(self.heading_rad)],
            [math.sin(self.heading_rad),  math.cos(self.heading_rad)]
        ])
        self.world_corners = np.dot(self._local_corners, R.T) + self.center

    def get_axes(self):
        """获取OBB的两个主轴方向向量 (单位向量)"""
        ax1 = np.array([math.cos(self.heading_rad), math.sin(self.heading_rad)])
        ax2 = np.array([-math.sin(self.heading_rad), math.cos(self.heading_rad)])
        return [ax1, ax2]

def check_obb_intersection_sat(obb1: OBB, obb2: OBB):
    """
    使用分离轴定理 (SAT) 检查两个OBB是否碰撞。
    """
    axes = obb1.get_axes() + obb2.get_axes()
    
    for axis in axes:
        # 投影OBB1
        proj1 = [np.dot(corner, axis) for corner in obb1.world_corners]
        min1, max1 = min(proj1), max(proj1)
        
        # 投影OBB2
        proj2 = [np.dot(corner, axis) for corner in obb2.world_corners]
        min2, max2 = min(proj2), max(proj2)

        # 检查投影是否重叠
        if max1 < min2 or max2 < min1:
            return False  # 找到分离轴，不碰撞
            
    return True # 没有找到分离轴，发生碰撞

import logging
import os # 用于创建日志文件夹
from datetime import datetime # 用于生成带时间戳的日志文件名

# --- 在类的外部或顶部配置基础的 Logger ---
# 可以创建一个专门的函数来设置logger，以便在其他地方复用

def setup_logger(logger_name, log_file, level=logging.INFO):
    """设置一个logger实例"""
    # 创建logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level) # 设置logger的级别，低于此级别的日志不会被处理

    # 创建formatter，定义日志输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # 创建文件handler，用于写入日志文件
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    file_handler = logging.FileHandler(log_file, mode='a') # 'a' for append, 'w' for overwrite
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
        
    # 防止日志重复输出 (如果多次调用 setup_logger)
    logger.propagate = False 

    return logger

class AStarNode:
    def __init__(self, pos, heading, g_cost, h_cost, parent=None, speed=0.0, timestamp=0.0):
        self.pos = np.array(pos) # [x, y]
        self.heading = heading   # rad
        self.g_cost = g_cost     # 从起点到此节点的实际代价
        self.h_cost = h_cost     # 从此节点到目标点的启发式代价
        self.f_cost = g_cost + h_cost
        self.parent = parent
        self.speed = speed       # 节点关联的速度 (可选，但有用)
        self.timestamp = timestamp # 节点关联的时间戳 (用于与动态障碍物比较)

    def __lt__(self, other): # heapq需要这个来进行优先队列排序
        return self.f_cost < other.f_cost

    def __eq__(self, other): # 用于在closed_set中查找
        if not isinstance(other, AStarNode):
            return NotImplemented
        # 为了简化，这里仅比较位置（可能需要更精确的比较，如加入heading离散化）
        return np.allclose(self.pos, other.pos, atol=0.1) # 位置容差0.1m

    def __hash__(self): # 用于在closed_set中查找
        # 将位置离散化以便哈希
        return hash((round(self.pos[0], 1), round(self.pos[1], 1)))

class AStarPlanner:
    def __init__(self, bicycle_model, ego_vehicle_shape, forecast_dt):
        self._bicycle_model = bicycle_model
        self.ego_vehicle_shape = ego_vehicle_shape # {'length': L, 'width': W}
        self.forecast_dt = forecast_dt # 与主模型中的预测步长一致

        # A* 参数
        self.steering_samples = [-0.6, -0.3, 0.0, 0.3, 0.6] # 离散的转向控制样本
        self.acceleration_samples = [-5.0, -2.5, 0.0, 2.5, 5.0]        # 离散的加速度 m/s^2 (负数为减速)
        self.prediction_step_duration = 0.5 # A*中每一步模拟多长时间 (s)
        self.goal_tolerance = 1.5           # (m) 到达目标的容忍距离
        self.obstacle_check_radius = 5.0    # (m) 只检查此半径内的障碍物以提高效率
        self.max_planning_time_steps = 50   # A*搜索的最大时间步数，防止无限循环
        self.path_weight_lane_deviation = 0.2 # 偏离参考线代价权重
        self.path_weight_length = 1.0       # 路径长度代价权重
        self.path_weight_reverse = 100.0    # 倒车惩罚 (如果允许)

    def _heuristic(self, pos1, pos2):
        # 欧氏距离启发函数
        return np.linalg.norm(pos1 - pos2)

    def _get_obstacle_obbs_at_time(self, predicted_other_trajectories, time_s):
        """
        获取在特定时间点所有障碍物的OBB列表。
        predicted_other_trajectories: list of lists, 每个子list是 [(pos, heading, speed, width, length), ...]
        """
        obstacle_obbs = []
        for traj in predicted_other_trajectories:
            # 找到最接近 time_s 的预测点
            # 简单实现：假设每个预测点间隔 self.forecast_dt
            idx = min(int(time_s / self.forecast_dt), len(traj) - 1)
            if idx < 0: continue

            obs_pos, obs_heading, _, obs_width, obs_length = traj[idx]
            obstacle_obbs.append(OBB(obs_pos, obs_width, obs_length, obs_heading))
        return obstacle_obbs

    def _is_safe_state(self, ego_pos, ego_heading, timestamp, predicted_other_trajectories):
        """检查当前ego状态在特定时间是否与任何障碍物碰撞"""
        ego_obb = OBB(ego_pos, self.ego_vehicle_shape['width'], self.ego_vehicle_shape['length'], ego_heading)
        obstacle_obbs_at_t = self._get_obstacle_obbs_at_time(predicted_other_trajectories, timestamp)
        for obs_obb in obstacle_obbs_at_t:
            # 粗略距离筛选
            if np.linalg.norm(ego_pos - obs_obb.center) > \
               (self.ego_vehicle_shape['length']/2 + obs_obb.length/2 + 5.0): # 加一个buffer
                continue
            if check_obb_intersection_sat(ego_obb, obs_obb):
                return False
        return True
    
    def _get_cost_of_arc(self, parent_node, child_pos, child_heading, ref_lane_path_for_astar=None):
        """计算从父节点到子节点的g_cost增量"""
        cost = 0
        # 1. 路径长度代价
        path_segment_length = np.linalg.norm(child_pos - parent_node.pos)
        cost += self.path_weight_length * path_segment_length

        # 2. 偏离参考路径的代价 (可选，但推荐)
        if ref_lane_path_for_astar:
            # 找到child_pos在参考路径上的投影点，计算横向误差
            # 简化：假设ref_lane_path_for_astar是一系列点，找到最近点
            # 实际应用中，应该使用车道对象的 local_coordinates 方法
            min_dist_to_ref = float('inf')
            # 这是一个简化的查找，真实场景中应该用更高效的投影
            # for pt_on_ref in ref_lane_path_for_astar.points: # 假设 ref_lane_path_for_astar 是 Lane 对象
            #     dist = np.linalg.norm(child_pos - pt_on_ref)
            #     if dist < min_dist_to_ref:
            #         min_dist_to_ref = dist
            # cost += self.path_weight_lane_deviation * min_dist_to_ref
            
            # 使用 Lane.local_coordinates 来获取横向偏移
            try:
                _, lat_dist = ref_lane_path_for_astar.local_coordinates(child_pos)
                cost += self.path_weight_lane_deviation * abs(lat_dist)
            except: # 如果点离车道太远，local_coordinates 可能失败
                cost += self.path_weight_lane_deviation * 10 # 较大惩罚


        # 3. 倒车惩罚 (如果自行车模型输出的速度是负的)
        # 假设自行车模型返回的速度总是正的，通过油门/刹车控制方向
        # 如果要考虑倒车，需要检查速度方向或朝向变化

        return cost


    def plan_path(self, start_pos, start_heading, start_speed, goal_pos,
                  predicted_other_trajectories, current_ref_lane):
        """
        执行A*搜索。
        start_pos, start_heading, start_speed: Ego车初始状态
        goal_pos: 规划的目标位置 [x,y]
        predicted_other_trajectories: 其他车辆的预测轨迹列表，
                                      每个元素是 [(pos, heading, speed, width, length), ...] 的序列
        current_ref_lane: 当前参考车道对象，用于计算偏离代价
        """
        open_set = []
        closed_set = set()

        start_node = AStarNode(start_pos, start_heading, 0, self._heuristic(start_pos, goal_pos), speed=start_speed, timestamp=0.0)
        heapq.heappush(open_set, start_node)

        path_found = None
        
        planning_steps_done = 0

        while open_set and planning_steps_done < self.max_planning_time_steps:
            planning_steps_done +=1
            current_node = heapq.heappop(open_set)

            if np.linalg.norm(current_node.pos - goal_pos) < self.goal_tolerance:
                path_found = []
                temp = current_node
                while temp:
                    path_found.append({'pos': temp.pos, 'heading': temp.heading, 'speed': temp.speed, 'time':temp.timestamp})
                    temp = temp.parent
                return path_found[::-1] # 返回逆序路径

            if current_node in closed_set: # 检查是否已在closed_set中（基于哈希和等于）
                continue
            closed_set.add(current_node)

            # 生成后继节点
            for steer in self.steering_samples:
                for accel_val in self.acceleration_samples: # accel_val 是实际加速度值
                    # 将加速度转换为油门/刹车信号 (这部分需要与您的车辆模型匹配)
                    # 简化: 直接使用accel_val, 假设自行车模型能处理
                    # 真实场景中: throttle_brake = self._convert_accel_to_action(accel_val, current_node.speed)
                    # 这里假设自行车模型直接使用加速度，或者需要调整
                    # throttle_brake = accel_val # 这可能不直接适用，取决于您的_bicycle_model.forward
                    
                    # 假设您的_bicycle_model.forward的控制输入是[steer, throttle, brake]
                    # 我们需要将accel_val转换为throttle或brake
                    throttle, brake = 0.0, 0.0
                    if accel_val > 0:
                        # 简化映射，假设最大加速度为5m/s^2时油门为1
                        throttle = np.clip(accel_val / 5.0, 0, 1)
                    elif accel_val < 0:
                        # 简化映射，假设最大减速度为-5m/s^2时刹车为1 (负号表示刹车)
                        brake = np.clip(accel_val / -5.0, 0, 1) # brake是正值

                    control = [steer, throttle, brake]
                    
                    # 使用自行车模型预测下一个状态
                    # 注意：_bicycle_model.forward 可能需要多次调用以达到 self.prediction_step_duration
                    num_sim_steps = int(self.prediction_step_duration / self._bicycle_model.dt)
                    next_pos, next_heading, next_speed = current_node.pos, current_node.heading, current_node.speed
                    
                    # 检查路径段的安全性
                    is_segment_safe = True
                    temp_path_segment_states = [] # 用于存储模拟的小步骤状态

                    for i in range(num_sim_steps):
                        # 传入控制：steer, throttle_brake (需要适配您的模型)
                        # 假设您的bicycle_model.forward输入是 [steer, throttle_value (0-1), brake_value (0-1)]
                        # control_for_model = [steer, accel_val if accel_val > 0 else 0, -accel_val if accel_val < 0 else 0]
                        
                        # 模拟一步
                        sim_next_pos, sim_next_heading, sim_next_speed = self._bicycle_model.forward(
                            next_pos, next_heading, next_speed, control
                        )
                        
                        current_sim_time = current_node.timestamp + (i + 1) * self._bicycle_model.dt
                        temp_path_segment_states.append((sim_next_pos, sim_next_heading, current_sim_time))

                        if not self._is_safe_state(sim_next_pos, sim_next_heading, current_sim_time, predicted_other_trajectories):
                            is_segment_safe = False
                            break # 这条子路径不安全
                        
                        next_pos, next_heading, next_speed = sim_next_pos, sim_next_heading, sim_next_speed
                    
                    if not is_segment_safe:
                        continue # 尝试下一个动作

                    # 如果安全，创建新节点
                    new_timestamp = current_node.timestamp + self.prediction_step_duration
                    
                    # 检查新节点是否与终态障碍物碰撞 (因为上面只检查了路径段)
                    if not self._is_safe_state(next_pos, next_heading, new_timestamp, predicted_other_trajectories):
                        continue

                    g_cost = current_node.g_cost + self._get_cost_of_arc(current_node, next_pos, next_heading, current_ref_lane)
                    h_cost = self._heuristic(next_pos, goal_pos)
                    neighbor_node = AStarNode(next_pos, next_heading, g_cost, h_cost, current_node, next_speed, new_timestamp)

                    # 检查是否已经在 closed_set 中 (基于位置的粗略检查)
                    # 一个更鲁棒的方法是使用离散化的状态 (pos_x_bin, pos_y_bin, heading_bin) 作为 closed_set 的键
                    already_closed = False
                    for closed_node_iter in closed_set: # Python set 不保证顺序，所以要迭代
                        if neighbor_node == closed_node_iter and neighbor_node.f_cost >= closed_node_iter.f_cost:
                            already_closed = True
                            break
                    if already_closed:
                        continue
                    
                    # 检查是否在 open_set 中且有更优路径
                    # heapq 不直接支持更新元素优先级，常见做法是直接添加，旧的会被后处理
                    # 或者，遍历open_set (效率较低)
                    
                    heapq.heappush(open_set, neighbor_node)
        return None # 未找到路径

import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
# import matplotlib
# matplotlib.use('TkAgg')

def visualize_trajectories(ego_trajectory, 
                           other_trajectories, 
                           colliding_vehicle_idx=-1, # 高亮显示您怀疑的特定车辆的索引
                           title="Trajectory Prediction Visualization",
                           dt=0.1): # 您预测的时间步长
    """
    可视化主车和背景车的预测轨迹。

    Args:
        ego_trajectory (list): 主车的预测轨迹，格式为 [(pos, heading, speed, width, length), ...]
        other_trajectories (list of lists): 背景车的预测轨迹
        colliding_vehicle_idx (int): 您怀疑会发生碰撞的背景车在 other_trajectories 中的索引，用于高亮显示。
        title (str): 图像标题。
        dt (float): 每个预测步的时间间隔（秒）。
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    num_frames = len(ego_trajectory)

    # --- 1. 绘制完整的轨迹路径（虚线）---
    ego_path_x = [p[0][0] for p in ego_trajectory]
    ego_path_y = [p[0][1] for p in ego_trajectory]
    ax.plot(ego_path_x, ego_path_y, 'b-', label="Ego Full Path", alpha=0.8)
    ax.plot(ego_path_x[0], ego_path_y[0], 'r.' , alpha=0.8)
    # ax.plot(ego_path_x[-1], ego_path_x[-1], 'g.', alpha=0.8)

    for i, other_traj in enumerate(other_trajectories):
        if not other_traj: continue
        other_path_x = [p[0][0] for p in other_traj]
        other_path_y = [p[0][1] for p in other_traj]
        color = 'r--' if i == colliding_vehicle_idx else 'k-'
        label = "Suspected Vehicle Full Path" if i == colliding_vehicle_idx else None
        ax.plot(other_path_x, other_path_y, color, label=label, alpha=0.8)
        ax.plot(other_path_x[0], other_path_y[0], 'r.' , alpha=0.8)
        # ax.plot(other_path_x[-1], other_path_y[-1], 'g.', alpha=0.8)

    # --- 设置图表样式 ---
    ax.set_xlabel("X coordinate (m)")
    ax.set_ylabel("Y coordinate (m)")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box') # 保证x,y轴比例相同，车辆不变形

    # --- 3. 创建并显示动画 ---
    plt.savefig('./logs/predict_collision.png')
    plt.close()


MAIN_POLICY_LOGGER_NAME = "MetaDriveMainPolicySafe"
LOG_FILENAME = "policy_run.log" # 或者带时间戳，但如果带时间戳，则每次运行程序都是新文件
LOG_FILE_PATH = os.path.join("logs/bctrans_expertAPv1_dagger_traj_target_navi_V4_MDWaymo-07", LOG_FILENAME)

# 确保主logger只被配置一次
main_policy_logger = setup_logger(MAIN_POLICY_LOGGER_NAME, LOG_FILE_PATH, level=logging.DEBUG)

class MetaDriveAutoPilotPolicy(BasePolicy):
    def __init__(self, control_object, random_seed):
        super(MetaDriveAutoPilotPolicy, self).__init__(control_object=control_object, random_seed=random_seed)

        # --- PID 控制器 (根据论文参数) ---
        # 横向 (转向) 控制器
        self._lateral_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        # 纵向 (速度) 控制器
        self._longitudinal_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

        # --- 运动学模型 ---
        self._bicycle_model = KinematicBicycleModel(dt=0.1) # 预测步长 100ms

        # --- 专家参数 (来自论文描述) ---
        self.target_speed_standard = 10.0  # m/s
        self.target_speed_intersection = 5.0 # m/s
        self.target_speed_halt = 0.0 # m/s

        self.min_waypoint_dist = 2 # m, 横向控制时，目标路径点至少这么远

        self.forecast_dt = 0.1 # s
        self.forecast_duration_intersection = 4.0 # s
        self.forecast_duration_other = 2 # s
        
        self.ego_collision_check_length_factor = 1  # 碰撞检测区域
        self.dynamic_speed_perception_radius = 20
        self.enable_matplotlib_debug_plot = False  # 添加一个开关来控制是否绘图
        self.plot_initialized = False
        self.fig = None
        self.ax = None
        self.forward_vehicles_speeds = []

        self.predict_other_v = False

        # --- A* 局部规划相关 ---
        self.local_planner_active = False
        self.current_local_path = None # 存储A*生成的路径点 [{'pos': [x,y], 'heading':h, 'speed':s, 'time': t}, ...]
        self.current_local_path_index = 0
        self.a_star_goal_distance = 15.0 # A*规划的目标点在参考路径上的前瞻距离 (m)
        
        # 获取车辆尺寸，A*规划器需要
        ego_shape = {'length': control_object.LENGTH, 'width': control_object.WIDTH}
        # 使用 self.forecast_dt 作为 A* 内部预测的 dt 似乎更合理
        self.a_star_planner = AStarPlanner(self._bicycle_model, ego_shape, self.forecast_dt) 
        
        self.last_action = [0.0, 0.0] # 用于平滑控制或调试

        log_filename = "policy_run.log" 
        log_file_path = os.path.join("logs", log_filename) # 将日志存放在 "logs" 子目录下
        logger_name = "AutoPilotPolicy"
        self.logger = main_policy_logger
        self.logger.info(f"MetaDriveAutoPilotPolicy initialized for object: {control_object.name if hasattr(control_object, 'name') else 'N/A'}")
        self.logger.info(f"Random seed: {random_seed}")


    def reset(self):
        super(MetaDriveAutoPilotPolicy, self).reset() # 调用父类的reset
        self._lateral_controller.reset()
        self._longitudinal_controller.reset()
        self.local_planner_active = False
        self.current_local_path = None
        self.current_local_path_index = 0
        self.last_action = [0.0, 0.0]
        self.logger.info("Policy has been reset.")
        if self.enable_matplotlib_debug_plot and self.plot_initialized:
            plt.close(self.fig)
            self.plot_initialized = False
            self.fig = None
            self.ax = None


    def act(self, *args, **kwargs):
        ego_vehicle = self.control_object
        ego_pos_2d = ego_vehicle.position
        ego_heading = wrap_to_pi(ego_vehicle.heading_theta)
        ego_speed_ms = ego_vehicle.speed
        
        # 更新 episode_step (假设 policy 可以从 kwargs 或 agent_manager 获取)
        # self.episode_step = self.engine.episode_step # 或者通过kwargs传入
        # 在Waymo环境中，通常有一个全局的step计数器
        # 确保 self.episode_step 在这里是最新的

        at_intersection = self._is_at_intersection(ego_vehicle)

        # 行为决策：预测违规 (红灯、碰撞)
        # _predict_infractions 现在可能触发A*规划
        predicted_infraction, infraction_type, infraction_object_id, dynamic_target_speed_from_pred = self._predict_infractions(
            ego_vehicle, ego_pos_2d, ego_heading, ego_speed_ms, at_intersection
        )
        
        current_target_speed = dynamic_target_speed_from_pred # 使用来自预测模块的动态速度

        is_emergency_brake = False
        if predicted_infraction:
            if infraction_type == "red_light":
                current_target_speed = self.target_speed_halt
                is_emergency_brake = True # 红灯视为紧急情况
            elif infraction_type == "collision_emergency":
                current_target_speed = self.target_speed_halt
                is_emergency_brake = True
            elif infraction_type == "rear_collision_accelerate":
                # 目标速度已经由 _predict_infractions 设置为加速目标
                # 不需要额外操作，纵向控制会处理
                self.logger.info(f"ACT: Responding to rear_collision_accelerate, target_speed: {current_target_speed:.2f}")
                pass 
            # "collision_astar_planned" 意味着A*正在工作，目标速度由 dynamic_target_speed_from_pred 决定
        
        # 如果不在任何特殊违规情况，并且在路口，且没有激活A*，则应用路口限速
        if not predicted_infraction and at_intersection and not self.local_planner_active and not self.forward_vehicles_speeds:
            current_target_speed = min(current_target_speed, self.target_speed_intersection)
        
        # 如果A*激活，目标速度主要由A*路径上的建议速度或前方动态速度决定
        if self.local_planner_active and self.current_local_path:
            # 可以考虑从A*路径点获取建议速度，如果A*输出速度的话
            # 否则， current_target_speed 维持 dynamic_target_speed_from_pred
            # 这里的dynamic_target_speed_from_pred已经考虑了前方车辆，是比较合理的
            pass

        throttle_brake = self._longitudinal_control(current_target_speed, ego_speed_ms, is_emergency_brake)
        steering = self._lateral_control(ego_pos_2d, ego_heading, ego_speed_ms)
        
        action = [steering, throttle_brake]
        self.action_info["action"] = action
        self.last_action = action
        self.logger.info(f"#### scenario {self.engine.current_seed}, step {self.engine.episode_step}, action {action}, current_speed {ego_speed_ms}, target_speed {current_target_speed} ####")
        return action

    def _is_at_intersection(self, vehicle_obj):
        """
        判断车辆是否在路口 (与之前实现类似, 基于前方路径曲率)
        """
        if not vehicle_obj.navigation:
            return False
        nav = vehicle_obj.navigation
        ego_pos = vehicle_obj.position
        current_lane = nav.current_ref_lanes[0]
        long, _ = current_lane.local_coordinates(ego_pos)
        
        # 检查前方路径的曲率
        points_for_curvature = []
        for i in range(3): # 取前方3个点，间隔可以调整
            points_for_curvature.append(current_lane.position(long + i * 2.0, 0)) # 间隔2米

        if len(points_for_curvature) < 3: return False # 路径太短

        p1, p2, p3 = points_for_curvature
        vec1 = p2 - p1
        vec2 = p3 - p2
        norm_v1 = vec1 / (np.linalg.norm(vec1) + 1e-6)
        norm_v2 = vec2 / (np.linalg.norm(vec2) + 1e-6)
        dot_product = np.dot(norm_v1, norm_v2)
        
        # 阈值可以调整, 点积越小，弯道越急
        return dot_product < 0.985
    
    def _lateral_control(self, ego_pos_2d, ego_heading, ego_speed_ms):
        if not self.control_object.navigation: return 0.0

        target_waypoint_pos = None
        target_waypoint_for_pid = None # PID控制器实际追踪的点

        if self.local_planner_active and self.current_local_path:
            # --- 追踪A*局部路径 ---
            if self.current_local_path_index < len(self.current_local_path):
                # 选择局部路径上的下一个点作为目标
                # 可以选择稍微远一点的点以获得更平滑的追踪
                lookahead_idx = self.current_local_path_index
                # 向前看几个点，或者直到点足够远
                for i in range(self.current_local_path_index, len(self.current_local_path)):
                    dist_to_wp = np.linalg.norm(self.current_local_path[i]['pos'] - ego_pos_2d)
                    if dist_to_wp > self.min_waypoint_dist: # 例如，至少3.5米远
                        lookahead_idx = i
                        break
                    if i == len(self.current_local_path) - 1: # 如果是最后一个点
                        lookahead_idx = i
                
                target_waypoint_pos = self.current_local_path[lookahead_idx]['pos']
                target_waypoint_for_pid = target_waypoint_pos

                # 检查是否到达当前局部路径段的末端附近
                dist_to_current_target = np.linalg.norm(target_waypoint_pos - ego_pos_2d)
                if dist_to_current_target < 2.0 or \
                   (self.current_local_path_index == lookahead_idx and dist_to_current_target < 1.0) : # 如果目标点很近
                    self.current_local_path_index = lookahead_idx + 1 # 移动到下一个路径点
                
                if self.current_local_path_index >= len(self.current_local_path):
                    self.logger.info("INFO: Local A* path finished.")
                    self.local_planner_active = False
                    self.current_local_path = None
                    self.current_local_path_index = 0
                    # 切换回常规路径追踪
                    # return 0.0 # 或者立即重新规划，但最好在下一个act循环中自然过渡
            else: # 索引超出，路径结束
                self.logger.info("INFO: Local A* path index out of bounds, finishing.")
                self.local_planner_active = False
                self.current_local_path = None
                self.current_local_path_index = 0
                # return 0.0
        
        # 如果不是局部规划模式，或者局部规划刚结束，则使用全局参考线
        if not self.local_planner_active or target_waypoint_for_pid is None:
            current_ref_lane = self.control_object.navigation.current_ref_lanes[0]
            current_long, _ = current_ref_lane.local_coordinates(ego_pos_2d)
            lookahead_dist = max(
                self.min_waypoint_dist,
                np.clip(ego_speed_ms * 0.3, self.min_waypoint_dist, 15.0)
            )
            target_waypoint_pos = current_ref_lane.position(current_long + lookahead_dist, 0)
            target_waypoint_for_pid = target_waypoint_pos

        if target_waypoint_for_pid is None:
            return 0.0

        vec_to_target = target_waypoint_for_pid - ego_pos_2d
        angle_to_target_math = math.atan2(vec_to_target[1], vec_to_target[0])
        angle_error = wrap_to_pi(angle_to_target_math - ego_heading)
        steering_output = self._lateral_controller.step(angle_error)
        
        # --- Matplotlib 调试绘图逻辑 (绘制A*路径) ---
        if self.enable_matplotlib_debug_plot and target_waypoint_for_pid is not None:
            if not self.plot_initialized:
                plt.ion()
                self.fig, self.ax = plt.subplots(figsize=(10, 10))
                self.plot_initialized = True
            
            self.ax.clear()
            self.ax.scatter(ego_pos_2d[0], ego_pos_2d[1], c='red', marker='o', s=100, label='Ego Vehicle')
            arrow_len = 2.0
            ego_dx = arrow_len * math.cos(ego_heading)
            ego_dy = arrow_len * math.sin(ego_heading)
            self.ax.arrow(ego_pos_2d[0], ego_pos_2d[1], ego_dx, ego_dy, head_width=0.5, head_length=0.7, fc='red', ec='red')

            self.ax.scatter(target_waypoint_for_pid[0], target_waypoint_for_pid[1], c='lime', marker='x', s=150, label='PID Target WP', zorder=10)
            self.ax.plot([ego_pos_2d[0], target_waypoint_for_pid[0]], 
                         [ego_pos_2d[1], target_waypoint_for_pid[1]], 
                         'b--', linewidth=1.0, label='Line to PID Target')

            # 绘制全局参考路径
            if self.control_object.navigation and self.control_object.navigation.current_ref_lanes:
                ref_lane = self.control_object.navigation.current_ref_lanes[0]
                lane_path_to_draw = ref_lane.segment_property # 通常是 {'points': [[x,y],...]}
                if 'points' in lane_path_to_draw:
                    pts = np.array(lane_path_to_draw['points'])
                    self.ax.plot(pts[:,0], pts[:,1], 'gray', linestyle=':', linewidth=1.0, label="Reference Lane")


            # 如果有A*局部路径，绘制它
            if self.local_planner_active and self.current_local_path:
                path_pts = np.array([wp['pos'] for wp in self.current_local_path])
                self.ax.plot(path_pts[:,0], path_pts[:,1], 'cyan', marker='.', markersize=5, linewidth=2.0, label='A* Local Path')
                if self.current_local_path_index < len(self.current_local_path):
                     curr_astar_target = self.current_local_path[self.current_local_path_index]['pos']
                     self.ax.scatter(curr_astar_target[0], curr_astar_target[1], c='magenta', marker='s', s=80, label='Current A* WP', zorder=9)


            # (可选) 绘制预测的障碍物轨迹 (需要从 _predict_infractions 中获取或重新计算)
            # all_vehicles_dict = self.engine.traffic_manager.current_traffic_data
            # other_vehicles_data = [ ... ] # 获取其他车辆数据
            # _, _, _, predicted_other_vehicle_trajectories = self._get_predicted_trajectories(...) # 需要一个辅助函数
            # for obs_traj in predicted_other_vehicle_trajectories:
            #     obs_pts_np = np.array([state[0] for state in obs_traj]) # state[0] is pos
            #     self.ax.plot(obs_pts_np[:,0], obs_pts_np[:,1], 'orange', marker='x', markersize=3, alpha=0.7, label="Predicted Obstacle Traj")


            plot_range = 40
            self.ax.set_xlim(ego_pos_2d[0] - plot_range, ego_pos_2d[0] + plot_range)
            self.ax.set_ylim(ego_pos_2d[1] - plot_range, ego_pos_2d[1] + plot_range)
            self.ax.set_xlabel("X (m)"); self.ax.set_ylabel("Y (m)")
            self.ax.set_title(f"Expert Policy Debug (Local Plan Active: {self.local_planner_active})")
            self.ax.legend(fontsize='small'); self.ax.grid(True); self.ax.set_aspect('equal', adjustable='box')
            plt.draw(); plt.pause(0.001)
            
        return np.clip(steering_output, -1.0, 1.0)

    
    def _longitudinal_control(self, target_speed_ms, current_speed_ms, is_emergency_brake):
        if is_emergency_brake: # 明确的紧急刹车信号
            self._longitudinal_controller.reset()
            return -1.0 # 最大刹车

        speed_error = target_speed_ms - current_speed_ms
        acceleration_command = self._longitudinal_controller.step(speed_error)
        
        if acceleration_command > 0:
            max_accel_estim = self.control_object.config.get("max_acceleration", 5.0)
            throttle = np.clip(acceleration_command / max_accel_estim, 0.0, 1.0)
            return throttle
        else:
            max_decel_estim = self.control_object.config.get("max_deceleration", 5.0) # 通常为正值
            brake_ratio = acceleration_command / (-max_decel_estim) # accel_cmd是负数  TODO
            brake = -np.clip(brake_ratio, 0.0, 1.0) # brake输出是[-1, 0]
            return brake
    
    def _get_raw_other_vehicle_predictions(self, ego_vehicle, num_forecast_steps):
        """
        获取其他车辆的预测轨迹（位置、朝向、速度、尺寸）。
        如果 predict_other_v 为 False，则使用Waymo真值。
        否则，使用运动学模型进行预测。
        返回: list of trajectories,每个轨迹是 [(pos, heading, speed, width, length), ...]
        """
        all_vehicles_dict = self.engine.traffic_manager.current_traffic_data
        other_vehicles_objects = [
            v_obj for v_id, v_obj in all_vehicles_dict.items()
            if v_id != self.engine.traffic_manager.sdc_track_index # 使用 sdc_track_index
        ]
        adv_name = self.engine.traffic_manager.adv_name
        adv_trajs = self.engine.traffic_manager.adv_traj
        predicted_other_trajectories = []

        current_episode_step = self.engine.episode_step # 获取当前仿真步数

        for other_v_data in other_vehicles_objects: # other_v_data 是Waymo场景中的车辆字典
            # 从Waymo数据中提取车辆尺寸 (如果没有，用ego车尺寸作为近似)
            # Waymo数据通常包含 length, width, height for each object
            # obj_id = ??? # 需要知道这个other_v_data对应的ID
            # type_specific_states = self.engine.data_manager.get_scenario(self.engine.current_map.scenario_id, should_copy=False)["tracks"][obj_id]["state"]
            # vehicle_length = type_specific_states['length'][current_episode_step]
            # vehicle_width = type_specific_states['width'][current_episode_step]
            # 简化：假设所有车辆尺寸与Ego类似，或从Waymo数据中提取
            # Waymo数据中 'state' 字段下有 'length', 'width'
            if other_v_data['type'] == 'PEDESTRIAN':  # TODO
                continue
            # 使用 try-except 来处理可能不存在的 key 或索引
            try:
                vehicle_length = other_v_data['state']['length'][current_episode_step] if 'length' in other_v_data['state'] else ego_vehicle.LENGTH
                vehicle_width  = other_v_data['state']['width'][current_episode_step] if 'width' in other_v_data['state'] else ego_vehicle.WIDTH
            except IndexError: # 如果 current_episode_step 超出范围 (不太可能在有效数据中)
                 vehicle_length = ego_vehicle.LENGTH
                 vehicle_width = ego_vehicle.WIDTH


            one_vehicle_traj = []
            if self.predict_other_v: # 自己预测 (这里需要实现之前您代码中的估计逻辑)
                if not other_v_data['state']['valid'][current_episode_step]:
                    predicted_other_trajectories.append([]) # 空轨迹
                    continue
                
                current_other_pos = np.array(other_v_data['state']['position'][current_episode_step,:2])
                current_other_heading = other_v_data['state']['heading'][current_episode_step]
                current_other_vel_xy = other_v_data['state']['velocity'][current_episode_step,:2]
                current_other_speed = np.linalg.norm(current_other_vel_xy)
                
                # --- 估计控制输入 (与您之前代码类似) ---
                estimated_steer, estimated_throttle, estimated_brake = 0.0, 0.0, 0.0
                if current_episode_step > 0:
                    prev_step = current_episode_step - 1
                    if other_v_data['state']['valid'][prev_step]:
                        prev_heading = other_v_data['state']['heading'][prev_step]
                        prev_vel_xy = other_v_data['state']['velocity'][prev_step,:2]
                        prev_speed = np.linalg.norm(prev_vel_xy)
                        
                        heading_change = wrap_to_pi(current_other_heading - prev_heading)
                        heading_rate = heading_change / self.engine.global_config["physics_world_step_size"] # 使用引擎dt

                        if abs(prev_speed) > 0.1:
                            # 简化转向估计 (实际可能需要更复杂的逆向自行车模型)
                            # tan(delta) = L * omega / v
                            # steer_angle = atan( L_rear_axle_to_cg * heading_rate / prev_speed)
                            # estimated_steer = steer_angle / max_steer_angle_of_model
                            # 粗略估计转向，假设与heading_rate成正比，需要校准
                            estimated_steer = np.clip(heading_rate * 0.2, -1.0, 1.0) # 0.2是经验系数

                        accel = (current_other_speed - prev_speed) / self.engine.global_config["physics_world_step_size"]
                        if accel > 0.1: # 加速阈值
                            estimated_throttle = np.clip(accel / 5.0, 0.0, 1.0) # 假设最大油门加速度5m/s^2
                        elif accel < -0.1: # 减速阈值
                            estimated_brake = np.clip(accel / -5.0, 0.0, 1.0) # 假设最大刹车减速度-5m/s^2
                
                other_control = [estimated_steer, estimated_throttle, estimated_brake]
                # --- 预测轨迹 ---
                temp_pos, temp_head, temp_speed = current_other_pos, current_other_heading, current_other_speed
                for _ in range(num_forecast_steps):
                    one_vehicle_traj.append((np.copy(temp_pos), temp_head, temp_speed, vehicle_width, vehicle_length))
                    next_pos, next_head, next_speed = self._bicycle_model.forward(
                        temp_pos, temp_head, temp_speed, other_control
                    )
                    temp_pos, temp_head, temp_speed = next_pos, next_head, next_speed
            
            else: # 使用Waymo真值轨迹
                if other_v_data['metadata']['object_id'] in adv_name and len(adv_trajs[adv_name.index(other_v_data['metadata']['object_id'])]):
                    adv_traj = adv_trajs[adv_name.index(other_v_data['metadata']['object_id'])]
                    for i in range(num_forecast_steps):
                        forecast_step_idx = i
                        # Waymo数据长度通常是91 (0到90)
                        if forecast_step_idx >= len(adv_traj):
                            # 如果超出数据范围或无效，可以复制最后一个有效状态或停止预测
                            if one_vehicle_traj: # 复制最后一个
                                last_valid_state = one_vehicle_traj[-1]
                                one_vehicle_traj.append(last_valid_state)
                            else: # 如果一开始就无效/超范围，则为空
                                break 
                            continue

                        pos = np.array(adv_traj[forecast_step_idx][0:2])
                        heading = adv_traj[forecast_step_idx][4]
                        vel_xy = adv_traj[forecast_step_idx][2:4]
                        speed = np.linalg.norm(vel_xy)
                        one_vehicle_traj.append((pos, heading, speed, vehicle_width, vehicle_length))   
                else: 
                    for i in range(num_forecast_steps):
                        forecast_step_idx = current_episode_step + i
                        # Waymo数据长度通常是91 (0到90)
                        if forecast_step_idx >= len(other_v_data['state']['position']) or \
                        not other_v_data['state']['valid'][forecast_step_idx]:
                            # 如果超出数据范围或无效，可以复制最后一个有效状态或停止预测
                            if one_vehicle_traj: # 复制最后一个
                                last_valid_state = one_vehicle_traj[-1]
                                one_vehicle_traj.append(last_valid_state)
                            else: # 如果一开始就无效/超范围，则为空
                                break 
                            continue

                        pos = np.array(other_v_data['state']['position'][forecast_step_idx, :2])
                        heading = other_v_data['state']['heading'][forecast_step_idx]
                        vel_xy = other_v_data['state']['velocity'][forecast_step_idx,:2]
                        speed = np.linalg.norm(vel_xy)
                        one_vehicle_traj.append((pos, heading, speed, vehicle_width, vehicle_length))
            
            predicted_other_trajectories.append(one_vehicle_traj)
        return predicted_other_trajectories
    
    def _get_collision_details(self, ego_state_at_collision, other_state_at_collision, ego_vehicle):
        """
        分析碰撞细节，判断碰撞类型（前、后、侧）。
        ego_state_at_collision: (pos, heading, speed, width, length)
        other_state_at_collision: (pos, heading, speed, width, length)
        Returns: str "front", "rear", "side_left", "side_right", "unknown"
                 float relative_speed_of_other (正数表示对方更快接近)
        """
        ego_pos, ego_heading, ego_speed, ego_w, ego_l = ego_state_at_collision
        other_pos, other_heading, other_speed, other_w, other_l = other_state_at_collision

        # 1. 计算相对位置（将other_pos转换到ego的局部坐标系）
        relative_pos = other_pos - ego_pos
        # 旋转到ego的局部坐标系
        cos_h, sin_h = math.cos(-ego_heading), math.sin(-ego_heading)
        local_other_x = relative_pos[0] * cos_h - relative_pos[1] * sin_h
        local_other_y = relative_pos[0] * sin_h + relative_pos[1] * cos_h

        # 2. 计算相对速度投影到ego朝向
        # ego速度向量
        ego_vel_vec = np.array([ego_speed * math.cos(ego_heading), ego_speed * math.sin(ego_heading)])
        # other速度向量
        other_vel_vec = np.array([other_speed * math.cos(other_heading), other_speed * math.sin(other_heading)])
        relative_vel_vec = other_vel_vec - ego_vel_vec
        
        # 相对速度在ego前进方向上的分量
        ego_direction_vec = np.array([math.cos(ego_heading), math.sin(ego_heading)])
        relative_speed_longitudinal = np.dot(relative_vel_vec, ego_direction_vec)
        # relative_speed_longitudinal > 0 表示对方在ego前进方向上比ego快（从后方接近或从前方远离得慢）
        # relative_speed_longitudinal < 0 表示ego在前进方向上比对方快（从后方远离或从前方接近得快）

        # 3. 判断碰撞类型
        # 简化判断：主要看 local_other_x 和 local_other_y
        # 这里的阈值需要根据车辆尺寸调整，增加一些buffer
        rear_threshold = -ego_l * 0.4 # 碰撞点在自车后部
        front_threshold = ego_l * 0.4 # 碰撞点在自车前部
        side_threshold = ego_w * 0.6 + other_w * 0.5 # 考虑两者宽度

        is_rear = local_other_x < rear_threshold and abs(local_other_y) < side_threshold
        is_front = local_other_x > front_threshold and abs(local_other_y) < side_threshold
        
        # 粗略判断侧面 (不在明确的前后，但在横向范围内)
        is_side = not is_rear and not is_front and abs(local_other_y) < side_threshold

        if is_rear:
            # 对于后方碰撞，如果对方在ego前进方向上比ego快 (relative_speed_longitudinal > 0)，则是典型的追尾
            # 如果对方比ego慢，可能是ego倒车撞后车，这在此场景不太可能
            return "rear", relative_speed_longitudinal 
        elif is_front:
            # 对于前方碰撞，如果对方在ego前进方向上比ego慢 (relative_speed_longitudinal < 0)，则是典型的追尾前方
            return "front", relative_speed_longitudinal
        elif is_side:
            # 侧面碰撞可以进一步区分为左侧或右侧
            if local_other_y > 0: # 对方在ego左侧
                return "side_left", relative_speed_longitudinal
            else: # 对方在ego右侧
                return "side_right", relative_speed_longitudinal
        
        return "unknown", relative_speed_longitudinal


    def _is_path_clear_ahead(self, ego_vehicle, ego_pos_2d, ego_heading_math, ego_speed_ms, 
                             predicted_other_vehicle_trajectories, check_duration_s=2.0):
        """
        检查自车前方一定时间内（例如2秒）路径是否清空，以便加速。
        只考虑自车正前方狭长区域内的障碍物。
        """
        num_check_steps = int(check_duration_s / self.forecast_dt)
        ego_w = ego_vehicle.WIDTH
        ego_l = ego_vehicle.LENGTH

        # 模拟自车在接下来 check_duration_s 内保持当前朝向和尝试加速的轨迹
        temp_ego_pos = np.copy(ego_pos_2d)
        temp_ego_heading = ego_heading_math
        temp_ego_speed = ego_speed_ms

        for i in range(num_check_steps):
            # 简单假设：自车尝试轻微加速或保持当前速度，不转向
            # 这里可以用更精确的预测，但为了快速检查，简化处理
            # 假设一个中等加速度
            sim_accel = 1.0 # m/s^2
            next_ego_speed = temp_ego_speed + sim_accel * self.forecast_dt
            dist_moved = (temp_ego_speed + next_ego_speed) / 2 * self.forecast_dt
            
            temp_ego_pos[0] += dist_moved * math.cos(temp_ego_heading)
            temp_ego_pos[1] += dist_moved * math.sin(temp_ego_heading)
            temp_ego_speed = next_ego_speed

            ego_check_obb = OBB(temp_ego_pos, ego_w, ego_l, temp_ego_heading)

            for other_v_traj in predicted_other_vehicle_trajectories:
                if i >= len(other_v_traj): continue

                other_pred_pos, other_pred_heading, _, other_w, other_l = other_v_traj[i]
                
                # 仅检查在自车前方一定范围内的车辆
                vec_to_other = other_pred_pos - temp_ego_pos
                dist_to_other = np.linalg.norm(vec_to_other)
                if dist_to_other > ego_l * 2 + other_l: # 粗筛，大于几倍车长就不细查了
                    continue

                angle_to_other = math.atan2(vec_to_other[1], vec_to_other[0])
                angle_diff = abs(wrap_to_pi(angle_to_other - temp_ego_heading))

                # 如果障碍物在自车前方很窄的角度内 (例如 +/- 15度)
                if angle_diff < math.radians(15):
                    other_obb = OBB(other_pred_pos, other_w, other_l, other_pred_heading)
                    if check_obb_intersection_sat(ego_check_obb, other_obb):
                        self.logger.info(f"INFO: Path ahead NOT clear due to object at step {i}")
                        return False # 前方有障碍
        self.logger.info("INFO: Path ahead IS clear.")
        return True # 前方路径清空

    def _predict_infractions(self, ego_vehicle, ego_pos_2d, ego_heading_math, ego_speed_ms, at_intersection):
        """
        预测未来是否会发生红灯或碰撞违规。
        返回 (bool: 是否有违规, str: 违规类型 "red_light" or "collision" or None)
        """
        # --- 1. 红灯预测 ---
        # MetaDrive可以直接获取交通灯信息
        for lane_info in ego_vehicle.navigation.current_ref_lanes:
            # PointLane 可能没有 traffic_light 属性，需要检查
            if hasattr(lane_info, 'traffic_light') and lane_info.traffic_light is not None:
                traffic_light = lane_info.traffic_light
                if traffic_light.get_state() == "red":
                    # 检查车辆是否在红灯的停止线附近
                    # 简单判断：如果红灯在前方一定距离内
                    # 这里的 stop_line_positions 是假设的，实际需要从lane_info获取
                    # 简化：如果红灯影响当前路径且很近
                    dist_to_light_stop_line = norm(traffic_light.position[0] - ego_pos_2d[0],
                                                   traffic_light.position[1] - ego_pos_2d[1]) # 粗略距离
                    if dist_to_light_stop_line < 15.0: # 15米内有红灯
                         # 还需要判断车辆是否正朝向红灯
                        vec_to_light = traffic_light.position - ego_pos_2d
                        angle_to_light = math.atan2(vec_to_light[1], vec_to_light[0])
                        if abs(wrap_to_pi(angle_to_light - ego_heading_math)) < math.pi / 4: #+/-45度内
                            return True, "red_light"
        
        # --- 2. 获取其他车辆的预测轨迹 ---
        forecast_duration = self.forecast_duration_intersection if at_intersection else self.forecast_duration_other
        num_forecast_steps = int(forecast_duration / self.forecast_dt)
        # predicted_other_vehicle_trajectories: list of lists, each sublist is [(pos, heading, speed, width, length), ...]
        predicted_other_vehicle_trajectories = self._get_raw_other_vehicle_predictions(ego_vehicle, num_forecast_steps)

        # --- 3. 动态目标速度计算 ---
        dynamic_target_speed = self.target_speed_standard # 默认值
        # 从 self._get_raw_other_vehicle_predictions 返回值中提取每个轨迹的第一个点（当前状态）
        self.forward_vehicles_speeds = []
        if ego_vehicle.navigation and ego_vehicle.navigation.current_ref_lanes:
            ego_lane = ego_vehicle.navigation.current_ref_lanes[0]
            ego_long_curr, _ = ego_lane.local_coordinates(ego_pos_2d)

            for other_traj in predicted_other_vehicle_trajectories:
                if not other_traj: continue # 空轨迹
                other_curr_pos = other_traj[0][0] # 当前位置
                other_curr_speed = other_traj[0][2] # 当前速度

                dist_to_other = np.linalg.norm(other_curr_pos - ego_pos_2d)
                if dist_to_other > self.dynamic_speed_perception_radius:
                    continue
                
                try:
                    other_long, other_lat = ego_lane.local_coordinates(other_curr_pos)
                    # 检查是否在同一车道 (或邻近)
                    if (abs(other_lat) < ego_lane.width * 1.2) & (other_curr_speed > 0): # 1.2倍车道宽度容差
                        self.forward_vehicles_speeds.append(other_curr_speed)
                except:
                    pass
        
        if self.forward_vehicles_speeds:
            calculated_speed = np.percentile(self.forward_vehicles_speeds, 75) # 取70分位数速度
            dynamic_target_speed = np.clip(calculated_speed, 
                                           self.target_speed_standard * 0.5,
                                        #    0, 
                                           self.target_speed_standard * 2.5) # 给定一个范围
            # dynamic_target_speed = min(dynamic_target_speed,max(forward_vehicles_speeds))
        else: # 前方无车或无有效速度
            dynamic_target_speed = self.target_speed_standard * 1.5 # 可略微提高

                
        # --- 4. 碰撞预测 ---
        # 4.1 预测Ego车轨迹 (假设按当前目标速度行驶，或尝试维持当前速度)
        ego_future_poses_kinematic = [] # list of (pos, heading_math, speed)
        current_ego_pos = np.copy(ego_pos_2d)
        current_ego_heading = ego_heading_math
        current_ego_speed = ego_speed_ms
        for _ in range(num_forecast_steps):
            # 用PID计算预测中的ego车控制
            pred_steering = self._lateral_control(current_ego_pos, current_ego_heading, current_ego_speed)
            # 假设预测时速度为当前的目标速度
            pred_speed_error = dynamic_target_speed - current_ego_speed 
            pred_accel_cmd = self._longitudinal_controller.step(pred_speed_error)
            pred_throttle = np.clip(pred_accel_cmd / 5.0, 0, 1) if pred_accel_cmd > 0 else 0
            # Ego车预测时刹车为0
            ego_pred_control = [pred_steering, pred_throttle, 0.0] 
            
            next_pos, next_heading, next_speed = self._bicycle_model.forward(
                current_ego_pos, current_ego_heading, current_ego_speed, ego_pred_control
            )
            ego_future_poses_kinematic.append((next_pos, next_heading, next_speed,ego_vehicle.WIDTH, ego_vehicle.LENGTH))
            current_ego_pos, current_ego_heading, current_ego_speed = next_pos, next_heading, next_speed

        # 4.2 碰撞检测
        imminent_collision_detected = False
        colliding_vehicle_data_at_collision = None # (pos, heading, speed, width, length)
        ego_state_at_collision = None
        collision_time_step = -1
        colliding_obj_original_idx = -1 # 用于从 predicted_other_vehicle_trajectories 中索引

        for i in range(num_forecast_steps): # 遍历每个预测时间步
            ego_pred_state_full = ego_future_poses_kinematic[i] # (pos, heading, speed, ego_w, ego_l)
            ego_pred_pos, ego_pred_heading, _, ego_w, ego_l = ego_pred_state_full
            ego_obb = OBB(ego_pred_pos, ego_w, ego_l, ego_pred_heading)

            for other_idx, other_v_traj in enumerate(predicted_other_vehicle_trajectories):
                if i >= len(other_v_traj) or not other_v_traj: continue 

                other_pred_state_full = other_v_traj[i] # (pos, heading, speed, other_w, other_l)
                other_pred_pos, other_pred_heading, _, other_w, other_l = other_pred_state_full
                
                if np.linalg.norm(ego_pred_pos - other_pred_pos) > (ego_l + other_l + 5.0): # 远距离粗筛
                    continue

                other_obb = OBB(other_pred_pos, other_w, other_l, other_pred_heading)

                if check_obb_intersection_sat(ego_obb, other_obb):
                    imminent_collision_detected = True
                    colliding_vehicle_data_at_collision = other_pred_state_full
                    ego_state_at_collision = ego_pred_state_full
                    collision_time_step = i
                    colliding_obj_original_idx = other_idx # 保存碰撞对象的索引
                    self.logger.info(f"INFO: Collision predicted at step {i} with an object (idx: {other_idx}).")
                    break 
            if imminent_collision_detected:
                break
        
        # visualize_trajectories(
        #         ego_trajectory=ego_future_poses_kinematic, # 使用保持速度的轨迹
        #         other_trajectories=predicted_other_vehicle_trajectories,
        #         colliding_vehicle_idx=3,
        #         title="[FAIL CASE] Constant Speed Prediction - No Collision Detected",
        #         dt=self.forecast_dt
        #     )

        # --- 4. 如果检测到碰撞 ---
        # --- 处理碰撞 ---
        if imminent_collision_detected:
            collision_type, rel_speed_long = self._get_collision_details(ego_state_at_collision, 
                                                                         colliding_vehicle_data_at_collision, 
                                                                         ego_vehicle)
            self.logger.info(f"INFO: Collision predicted type: {collision_type}, RelSpeedLong: {rel_speed_long:.2f} m/s, TimeStep: {collision_time_step}")

            # **特殊处理后方碰撞威胁**
            if collision_type == "rear" and rel_speed_long > 0.5: # 对方从后方快速接近 (阈值0.5m/s)
                # 检查前方是否安全以加速
                # 注意：predicted_other_vehicle_trajectories 此时包含所有车辆，包括可能在ego前方的
                # 我们需要排除掉当前正在从后方构成威胁的这辆车 (colliding_obj_original_idx)
                # 然后再检查前方是否有其他车辆阻挡
                # 实际上 _is_path_clear_ahead 应该能正确处理，它会检查所有其他车
                
                # 获取导致后方碰撞的车辆的当前速度
                rear_collider_current_speed = 0.0
                if colliding_obj_original_idx != -1 and predicted_other_vehicle_trajectories[colliding_obj_original_idx]:
                    rear_collider_current_speed = predicted_other_vehicle_trajectories[colliding_obj_original_idx][0][2] # 当前速度

                if self._is_path_clear_ahead(ego_vehicle, ego_pos_2d, ego_heading_math, ego_speed_ms, predicted_other_vehicle_trajectories):
                    self.logger.info("INFO: Rear collision predicted. Path ahead clear. Attempting to accelerate.")
                    self.local_planner_active = False # 如果之前有A*，取消它，优先纵向规避
                    self.current_local_path = None
                    # 目标速度：尝试匹配后方车辆速度或略快于它
                    if rear_collider_current_speed > ego_speed_ms:
                         # 目标是比后车快一点，或者达到一个较高的巡航速度
                        accelerate_target = max(rear_collider_current_speed + 2.0, self.target_speed_standard * 1.5)
                    else: # 后车不比我快，但我可能在减速，应保持或轻微加速
                        accelerate_target = max(ego_speed_ms + 1.0, self.target_speed_standard)

                    # 最终目标速度要考虑前方车辆（dynamic_target_speed已经计算了）
                    # final_target_speed = min(accelerate_target, dynamic_target_speed if forward_vehicles_speeds else accelerate_target) # 如果前方没车，dynamic_target_speed可能是标准速度，可以适当再快点
                    # final_target_speed = max(final_target_speed, ego_speed_ms) # 至少保持当前速度
                    if self.forward_vehicles_speeds:
                        final_target_speed = max(accelerate_target,(accelerate_target+dynamic_target_speed)/2)
                    else:
                        final_target_speed = accelerate_target


                    self.logger.info(f"INFO: Rear Evade. EgoSpd:{ego_speed_ms:.1f}, RearColliderSpd:{rear_collider_current_speed:.1f}, AccelTarget:{accelerate_target:.1f}, DynTarget:{dynamic_target_speed:.1f}, FinalTarget:{final_target_speed:.1f}")

                    return True, "rear_collision_accelerate", colliding_obj_original_idx, final_target_speed
                else: # 前方不安全，不能单纯加速
                    self.logger.info("WARN: Rear collision predicted, but path ahead is NOT clear. Falling back to A* or emergency.")
                    return True, "rear_collision_accelerate", colliding_obj_original_idx, rear_collider_current_speed + 1.0  # TODO
                    # 这种情况下，A*可能仍是最佳选择，或者如果A*也解决不了，就是紧急情况
            else:
                return True, "collision", colliding_obj_original_idx, self.target_speed_halt  # TODO

            # # 如果不是需要特殊处理的后方碰撞，或者后方碰撞但前方不安全，则走标准A*流程
            # if not self.local_planner_active:
            #     self.logger.info("INFO: Imminent collision (not rear-accelerate type or front blocked). Attempting A* local planning...")
            #     # ... (A*规划逻辑，与之前类似) ...
            #     # A* 目标点...
            #     if not ego_vehicle.navigation or not ego_vehicle.navigation.current_ref_lanes:
            #         self.logger.info("WARN: A* planning: No navigation path available for goal setting.")
            #         return True, "collision_emergency", colliding_obj_original_idx, self.target_speed_halt

            #     current_ref_lane = ego_vehicle.navigation.current_ref_lanes[0]
            #     current_long, _ = current_ref_lane.local_coordinates(ego_pos_2d) # 使用当前ego位置
            #     goal_pos_astar = current_ref_lane.position(current_long + self.a_star_goal_distance, 0)
            
            #     planned_path = self.a_star_planner.plan_path(
            #         ego_pos_2d, ego_heading_math, ego_speed_ms, # 当前状态作为起点
            #         goal_pos_astar,
            #         predicted_other_vehicle_trajectories,
            #         current_ref_lane
            #     )
            #     if planned_path and len(planned_path) > 1:
            #         self.logger.info(f"INFO: A* planner found a path with {len(planned_path)} waypoints.")
            #         self.current_local_path = planned_path
            #         self.current_local_path_index = 0 
            #         self.local_planner_active = True
            #         # A*规划成功后，目标速度应由A*路径本身引导，或使用当前的dynamic_target_speed
            #         # 暂时保持 dynamic_target_speed，因为A*主要负责横向
            #         return False, "collision_astar_planned", None, dynamic_target_speed 
            #     else:
            #         self.logger.info("WARN: A* planner failed to find a safe path. Emergency braking.")
            #         return True, "collision_emergency", colliding_obj_original_idx, self.target_speed_halt
            
            # else: # self.local_planner_active is True, 但仍然预测碰撞
            #       # 这意味着当前的A*路径仍然不安全
            #     self.logger.info(f"WARN: Collision predicted WHILE local A* path is active. Re-evaluating. Type: {collision_type}")
            #     # 针对这种情况，如果又是后方碰撞且前方清晰，也应该优先加速
            #     if collision_type == "rear" and rel_speed_long > 0.5 and \
            #        self._is_path_clear_ahead(ego_vehicle, ego_pos_2d, ego_heading_math, ego_speed_ms, predicted_other_vehicle_trajectories):
            #         self.logger.info("INFO: Rear collision during A*. Path ahead clear. Cancelling A* and accelerating.")
            #         self.local_planner_active = False 
            #         self.current_local_path = None
            #         rear_collider_current_speed = predicted_other_vehicle_trajectories[colliding_obj_original_idx][0][2]
            #         # (与上面的加速逻辑类似)
            #         if rear_collider_current_speed > ego_speed_ms:
            #             accelerate_target = max(rear_collider_current_speed + 2.0, self.target_speed_standard * 1.5)
            #         else:
            #             accelerate_target = max(ego_speed_ms + 1.0, self.target_speed_standard)
            #         # final_target_speed = min(accelerate_target, dynamic_target_speed if forward_vehicles_speeds else accelerate_target*1.2)
            #         final_target_speed = max(accelerate_target, ego_speed_ms)
            #         self.logger.info(f"INFO: Rear Evade (during A*). EgoSpd:{ego_speed_ms:.1f}, RearColliderSpd:{rear_collider_current_speed:.1f}, AccelTarget:{accelerate_target:.1f}, DynTarget:{dynamic_target_speed:.1f}, FinalTarget:{final_target_speed:.1f}")
            #         return True, "rear_collision_accelerate", colliding_obj_original_idx, final_target_speed
            #     else: # A*路径不安全，且不是可简单加速的后方碰撞，尝试重新规划或紧急刹车
            #         self.logger.info("WARN: A* path unsafe, not simple rear-accelerate. Attempting A* replan or emergency brake.")
            #         self.local_planner_active = False # 取消当前A*路径
            #         self.current_local_path = None
            #         # 重新尝试A*（可能会导致循环，但给一次机会）
            #         # (重复A*规划逻辑)
            #         if not ego_vehicle.navigation or not ego_vehicle.navigation.current_ref_lanes:
            #             return True, "collision_emergency", colliding_obj_original_idx, self.target_speed_halt
            #         current_ref_lane = ego_vehicle.navigation.current_ref_lanes[0]
            #         current_long, _ = current_ref_lane.local_coordinates(ego_pos_2d)
            #         goal_pos_astar = current_ref_lane.position(current_long + self.a_star_goal_distance, 0)
            #         planned_path = self.a_star_planner.plan_path(
            #             ego_pos_2d, ego_heading_math, ego_speed_ms, goal_pos_astar,
            #             predicted_other_vehicle_trajectories, current_ref_lane)
            #         if planned_path and len(planned_path) > 1:
            #             self.current_local_path = planned_path; self.current_local_path_index = 0; self.local_planner_active = True
            #             return False, "collision_astar_planned", None, dynamic_target_speed
            #         else: # 重规划也失败
            #             self.logger.info("A* planner: No path found or max steps reached.")
            #             return True, "collision_emergency", colliding_obj_original_idx, self.target_speed_halt

        return False, None, None, dynamic_target_speed
    
    def _lateral_control_for_prediction(self, ego_pos_2d, ego_heading, ego_speed_ms, ego_vehicle_obj):
        """专门为轨迹预测使用的简化横向控制，不修改主PID状态。"""
        if not ego_vehicle_obj.navigation or not ego_vehicle_obj.navigation.current_ref_lanes:
            return 0.0

        current_ref_lane = ego_vehicle_obj.navigation.current_ref_lanes[0]
        current_long, _ = current_ref_lane.local_coordinates(ego_pos_2d)
        
        lookahead_dist = max(
            self.min_waypoint_dist,
            np.clip(ego_speed_ms * 1.5, self.min_waypoint_dist, 15.0)
        )
        target_waypoint_pos = current_ref_lane.position(current_long + lookahead_dist, 0)

        if target_waypoint_pos is None: return 0.0

        vec_to_target = target_waypoint_pos - ego_pos_2d
        angle_to_target_math = math.atan2(vec_to_target[1], vec_to_target[0])
        angle_error = wrap_to_pi(angle_to_target_math - ego_heading)
 
        steering_output = self._lateral_controller.step(angle_error)
        return np.clip(steering_output, -1.0, 1.0)