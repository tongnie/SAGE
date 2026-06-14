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
    def __init__(self, dt=0.1):
        self.dt = dt

        self.front_wb = 1.2
        self.rear_wb = 1.3




        # self.rear_wb = 1.4178275






        # self.L_f = self.wheelbase * 0.5
        # self.L_r = self.wheelbase * 0.5


        # self.L_f = L / 2
        # self.L_r = L / 2




        self.CARLA_WB_PARAM_REAR = 1.4178275
        self.CARLA_WB_PARAM_TOTAL_INV = 1.0 / (self.CARLA_WB_PARAM_REAR - 0.090769015)

        self.steer_gain = 0.36848336
        self.brake_accel = -4.952399
        self.throt_accel = 0.5633837

    def forward(self, pos, heading_rad, speed_m_s, control_action):
        steer_norm, throttle_norm, brake_flag = control_action


        wheel_angle_rad = self.steer_gain * steer_norm

        if brake_flag > 0.5:
            acceleration = self.brake_accel
        else:
            acceleration = self.throt_accel * throttle_norm


        # beta = math.atan( (self.L_r / (self.L_f + self.L_r)) * math.tan(wheel_angle_rad) )

        beta = math.atan( self.CARLA_WB_PARAM_REAR * self.CARLA_WB_PARAM_TOTAL_INV * math.tan(wheel_angle_rad) )

        next_pos_x = pos[0] + speed_m_s * math.cos(heading_rad + beta) * self.dt
        next_pos_y = pos[1] + speed_m_s * math.sin(heading_rad + beta) * self.dt

        # next_heading_rad = heading_rad + (speed_m_s / self.L_r) * math.sin(beta) * self.dt

        next_heading_rad = heading_rad + (speed_m_s / self.CARLA_WB_PARAM_REAR) * math.sin(beta) * self.dt
        next_heading_rad = wrap_to_pi(next_heading_rad)

        next_speed_m_s = speed_m_s + acceleration * self.dt
        next_speed_m_s = max(0.0, next_speed_m_s)

        return np.array([next_pos_x, next_pos_y]), next_heading_rad, next_speed_m_s


class OBB:
    def __init__(self, center_xy, width, length, heading_rad):
        self.center = np.array(center_xy)
        self.width = width
        self.length = length
        self.heading_rad = heading_rad


        self.half_width = self.width / 2.0
        self.half_length = self.length / 2.0


        # local_corners: (front_left, front_right, rear_right, rear_left)
        self._local_corners = np.array([
            [-self.half_length,  self.half_width],
            [ self.half_length,  self.half_width],
            [ self.half_length, -self.half_width],
            [-self.half_length, -self.half_width]
        ])


        R = np.array([
            [math.cos(self.heading_rad), -math.sin(self.heading_rad)],
            [math.sin(self.heading_rad),  math.cos(self.heading_rad)]
        ])
        self.world_corners = np.dot(self._local_corners, R.T) + self.center

    def get_axes(self):
        ax1 = np.array([math.cos(self.heading_rad), math.sin(self.heading_rad)])
        ax2 = np.array([-math.sin(self.heading_rad), math.cos(self.heading_rad)])
        return [ax1, ax2]

def check_obb_intersection_sat(obb1: OBB, obb2: OBB):
    axes = obb1.get_axes() + obb2.get_axes()

    for axis in axes:

        proj1 = [np.dot(corner, axis) for corner in obb1.world_corners]
        min1, max1 = min(proj1), max(proj1)


        proj2 = [np.dot(corner, axis) for corner in obb2.world_corners]
        min2, max2 = min(proj2), max(proj2)


        if max1 < min2 or max2 < min1:
            return False

    return True

import logging
import os
from datetime import datetime




def setup_logger(logger_name, log_file, level=logging.INFO):

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)


    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')



    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_handler = logging.FileHandler(log_file, mode='a') # 'a' for append, 'w' for overwrite
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


    logger.propagate = False

    return logger

class AStarNode:
    def __init__(self, pos, heading, g_cost, h_cost, parent=None, speed=0.0, timestamp=0.0):
        self.pos = np.array(pos) # [x, y]
        self.heading = heading   # rad
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent
        self.speed = speed
        self.timestamp = timestamp

    def __lt__(self, other):
        return self.f_cost < other.f_cost

    def __eq__(self, other):
        if not isinstance(other, AStarNode):
            return NotImplemented

        return np.allclose(self.pos, other.pos, atol=0.1)

    def __hash__(self):

        return hash((round(self.pos[0], 1), round(self.pos[1], 1)))

class AStarPlanner:
    def __init__(self, bicycle_model, ego_vehicle_shape, forecast_dt):
        self._bicycle_model = bicycle_model
        self.ego_vehicle_shape = ego_vehicle_shape # {'length': L, 'width': W}
        self.forecast_dt = forecast_dt


        self.steering_samples = [-0.6, -0.3, 0.0, 0.3, 0.6]
        self.acceleration_samples = [-5.0, -2.5, 0.0, 2.5, 5.0]
        self.prediction_step_duration = 0.5
        self.goal_tolerance = 1.5
        self.obstacle_check_radius = 5.0
        self.max_planning_time_steps = 50
        self.path_weight_lane_deviation = 0.2
        self.path_weight_length = 1.0
        self.path_weight_reverse = 100.0

    def _heuristic(self, pos1, pos2):

        return np.linalg.norm(pos1 - pos2)

    def _get_obstacle_obbs_at_time(self, predicted_other_trajectories, time_s):
        obstacle_obbs = []
        for traj in predicted_other_trajectories:


            idx = min(int(time_s / self.forecast_dt), len(traj) - 1)
            if idx < 0: continue

            obs_pos, obs_heading, _, obs_width, obs_length = traj[idx]
            obstacle_obbs.append(OBB(obs_pos, obs_width, obs_length, obs_heading))
        return obstacle_obbs

    def _is_safe_state(self, ego_pos, ego_heading, timestamp, predicted_other_trajectories):
        ego_obb = OBB(ego_pos, self.ego_vehicle_shape['width'], self.ego_vehicle_shape['length'], ego_heading)
        obstacle_obbs_at_t = self._get_obstacle_obbs_at_time(predicted_other_trajectories, timestamp)
        for obs_obb in obstacle_obbs_at_t:

            if np.linalg.norm(ego_pos - obs_obb.center) >\
               (self.ego_vehicle_shape['length']/2 + obs_obb.length/2 + 5.0):
                continue
            if check_obb_intersection_sat(ego_obb, obs_obb):
                return False
        return True

    def _get_cost_of_arc(self, parent_node, child_pos, child_heading, ref_lane_path_for_astar=None):
        cost = 0

        path_segment_length = np.linalg.norm(child_pos - parent_node.pos)
        cost += self.path_weight_length * path_segment_length


        if ref_lane_path_for_astar:



            min_dist_to_ref = float('inf')


            #     dist = np.linalg.norm(child_pos - pt_on_ref)
            #     if dist < min_dist_to_ref:
            #         min_dist_to_ref = dist
            # cost += self.path_weight_lane_deviation * min_dist_to_ref


            try:
                _, lat_dist = ref_lane_path_for_astar.local_coordinates(child_pos)
                cost += self.path_weight_lane_deviation * abs(lat_dist)
            except:
                cost += self.path_weight_lane_deviation * 10






        return cost


    def plan_path(self, start_pos, start_heading, start_speed, goal_pos,
                  predicted_other_trajectories, current_ref_lane):
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
                return path_found[::-1]

            if current_node in closed_set:
                continue
            closed_set.add(current_node)


            for steer in self.steering_samples:
                for accel_val in self.acceleration_samples:








                    throttle, brake = 0.0, 0.0
                    if accel_val > 0:

                        throttle = np.clip(accel_val / 5.0, 0, 1)
                    elif accel_val < 0:

                        brake = np.clip(accel_val / -5.0, 0, 1)

                    control = [steer, throttle, brake]



                    num_sim_steps = int(self.prediction_step_duration / self._bicycle_model.dt)
                    next_pos, next_heading, next_speed = current_node.pos, current_node.heading, current_node.speed


                    is_segment_safe = True
                    temp_path_segment_states = []

                    for i in range(num_sim_steps):


                        # control_for_model = [steer, accel_val if accel_val > 0 else 0, -accel_val if accel_val < 0 else 0]


                        sim_next_pos, sim_next_heading, sim_next_speed = self._bicycle_model.forward(
                            next_pos, next_heading, next_speed, control
                        )

                        current_sim_time = current_node.timestamp + (i + 1) * self._bicycle_model.dt
                        temp_path_segment_states.append((sim_next_pos, sim_next_heading, current_sim_time))

                        if not self._is_safe_state(sim_next_pos, sim_next_heading, current_sim_time, predicted_other_trajectories):
                            is_segment_safe = False
                            break

                        next_pos, next_heading, next_speed = sim_next_pos, sim_next_heading, sim_next_speed

                    if not is_segment_safe:
                        continue


                    new_timestamp = current_node.timestamp + self.prediction_step_duration


                    if not self._is_safe_state(next_pos, next_heading, new_timestamp, predicted_other_trajectories):
                        continue

                    g_cost = current_node.g_cost + self._get_cost_of_arc(current_node, next_pos, next_heading, current_ref_lane)
                    h_cost = self._heuristic(next_pos, goal_pos)
                    neighbor_node = AStarNode(next_pos, next_heading, g_cost, h_cost, current_node, next_speed, new_timestamp)



                    already_closed = False
                    for closed_node_iter in closed_set:
                        if neighbor_node == closed_node_iter and neighbor_node.f_cost >= closed_node_iter.f_cost:
                            already_closed = True
                            break
                    if already_closed:
                        continue





                    heapq.heappush(open_set, neighbor_node)
        return None

import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
# import matplotlib
# matplotlib.use('TkAgg')

def visualize_trajectories(ego_trajectory,
                           other_trajectories,
                           colliding_vehicle_idx=-1,
                           title="Trajectory Prediction Visualization",
                           dt=0.1):
    fig, ax = plt.subplots(figsize=(12, 12))
    num_frames = len(ego_trajectory)


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


    ax.set_xlabel("X coordinate (m)")
    ax.set_ylabel("Y coordinate (m)")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')


    plt.savefig('./logs/predict_collision.png')
    plt.close()


MAIN_POLICY_LOGGER_NAME = "MetaDriveMainPolicySafe"
LOG_FILENAME = "policy_run.log"
LOG_FILE_PATH = os.path.join("logs/bctrans_expertAPv1_dagger_traj_target_navi_V4_MDWaymo-07", LOG_FILENAME)


main_policy_logger = setup_logger(MAIN_POLICY_LOGGER_NAME, LOG_FILE_PATH, level=logging.DEBUG)

class MetaDriveAutoPilotPolicy(BasePolicy):
    def __init__(self, control_object, random_seed):
        super(MetaDriveAutoPilotPolicy, self).__init__(control_object=control_object, random_seed=random_seed)



        self._lateral_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)

        self._longitudinal_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)


        self._bicycle_model = KinematicBicycleModel(dt=0.1)


        self.target_speed_standard = 10.0  # m/s
        self.target_speed_intersection = 5.0 # m/s
        self.target_speed_halt = 0.0 # m/s

        self.min_waypoint_dist = 2

        self.forecast_dt = 0.1 # s
        self.forecast_duration_intersection = 4.0 # s
        self.forecast_duration_other = 2 # s

        self.ego_collision_check_length_factor = 1
        self.dynamic_speed_perception_radius = 20
        self.enable_matplotlib_debug_plot = False
        self.plot_initialized = False
        self.fig = None
        self.ax = None
        self.forward_vehicles_speeds = []

        self.predict_other_v = False


        self.local_planner_active = False
        self.current_local_path = None
        self.current_local_path_index = 0
        self.a_star_goal_distance = 15.0


        ego_shape = {'length': control_object.LENGTH, 'width': control_object.WIDTH}

        self.a_star_planner = AStarPlanner(self._bicycle_model, ego_shape, self.forecast_dt)

        self.last_action = [0.0, 0.0]

        log_filename = "policy_run.log"
        log_file_path = os.path.join("logs", log_filename)
        logger_name = "AutoPilotPolicy"
        self.logger = main_policy_logger
        self.logger.info(f"MetaDriveAutoPilotPolicy initialized for object: {control_object.name if hasattr(control_object, 'name') else 'N/A'}")
        self.logger.info(f"Random seed: {random_seed}")


    def reset(self):
        super(MetaDriveAutoPilotPolicy, self).reset()
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






        at_intersection = self._is_at_intersection(ego_vehicle)



        predicted_infraction, infraction_type, infraction_object_id, dynamic_target_speed_from_pred = self._predict_infractions(
            ego_vehicle, ego_pos_2d, ego_heading, ego_speed_ms, at_intersection
        )

        current_target_speed = dynamic_target_speed_from_pred

        is_emergency_brake = False
        if predicted_infraction:
            if infraction_type == "red_light":
                current_target_speed = self.target_speed_halt
                is_emergency_brake = True
            elif infraction_type == "collision_emergency":
                current_target_speed = self.target_speed_halt
                is_emergency_brake = True
            elif infraction_type == "rear_collision_accelerate":


                self.logger.info(f"ACT: Responding to rear_collision_accelerate, target_speed: {current_target_speed:.2f}")
                pass



        if not predicted_infraction and at_intersection and not self.local_planner_active and not self.forward_vehicles_speeds:
            current_target_speed = min(current_target_speed, self.target_speed_intersection)


        if self.local_planner_active and self.current_local_path:



            pass

        throttle_brake = self._longitudinal_control(current_target_speed, ego_speed_ms, is_emergency_brake)
        steering = self._lateral_control(ego_pos_2d, ego_heading, ego_speed_ms)

        action = [steering, throttle_brake]
        self.action_info["action"] = action
        self.last_action = action
        self.logger.info(f"#### scenario {self.engine.current_seed}, step {self.engine.episode_step}, action {action}, current_speed {ego_speed_ms}, target_speed {current_target_speed} ####")
        return action

    def _is_at_intersection(self, vehicle_obj):
        if not vehicle_obj.navigation:
            return False
        nav = vehicle_obj.navigation
        ego_pos = vehicle_obj.position
        current_lane = nav.current_ref_lanes[0]
        long, _ = current_lane.local_coordinates(ego_pos)


        points_for_curvature = []
        for i in range(3):
            points_for_curvature.append(current_lane.position(long + i * 2.0, 0))

        if len(points_for_curvature) < 3: return False

        p1, p2, p3 = points_for_curvature
        vec1 = p2 - p1
        vec2 = p3 - p2
        norm_v1 = vec1 / (np.linalg.norm(vec1) + 1e-6)
        norm_v2 = vec2 / (np.linalg.norm(vec2) + 1e-6)
        dot_product = np.dot(norm_v1, norm_v2)


        return dot_product < 0.985

    def _lateral_control(self, ego_pos_2d, ego_heading, ego_speed_ms):
        if not self.control_object.navigation: return 0.0

        target_waypoint_pos = None
        target_waypoint_for_pid = None

        if self.local_planner_active and self.current_local_path:

            if self.current_local_path_index < len(self.current_local_path):


                lookahead_idx = self.current_local_path_index

                for i in range(self.current_local_path_index, len(self.current_local_path)):
                    dist_to_wp = np.linalg.norm(self.current_local_path[i]['pos'] - ego_pos_2d)
                    if dist_to_wp > self.min_waypoint_dist:
                        lookahead_idx = i
                        break
                    if i == len(self.current_local_path) - 1:
                        lookahead_idx = i

                target_waypoint_pos = self.current_local_path[lookahead_idx]['pos']
                target_waypoint_for_pid = target_waypoint_pos


                dist_to_current_target = np.linalg.norm(target_waypoint_pos - ego_pos_2d)
                if dist_to_current_target < 2.0 or\
                   (self.current_local_path_index == lookahead_idx and dist_to_current_target < 1.0) :
                    self.current_local_path_index = lookahead_idx + 1

                if self.current_local_path_index >= len(self.current_local_path):
                    self.logger.info("INFO: Local A* path finished.")
                    self.local_planner_active = False
                    self.current_local_path = None
                    self.current_local_path_index = 0


            else:
                self.logger.info("INFO: Local A* path index out of bounds, finishing.")
                self.local_planner_active = False
                self.current_local_path = None
                self.current_local_path_index = 0
                # return 0.0


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


            if self.control_object.navigation and self.control_object.navigation.current_ref_lanes:
                ref_lane = self.control_object.navigation.current_ref_lanes[0]
                lane_path_to_draw = ref_lane.segment_property
                if 'points' in lane_path_to_draw:
                    pts = np.array(lane_path_to_draw['points'])
                    self.ax.plot(pts[:,0], pts[:,1], 'gray', linestyle=':', linewidth=1.0, label="Reference Lane")



            if self.local_planner_active and self.current_local_path:
                path_pts = np.array([wp['pos'] for wp in self.current_local_path])
                self.ax.plot(path_pts[:,0], path_pts[:,1], 'cyan', marker='.', markersize=5, linewidth=2.0, label='A* Local Path')
                if self.current_local_path_index < len(self.current_local_path):
                     curr_astar_target = self.current_local_path[self.current_local_path_index]['pos']
                     self.ax.scatter(curr_astar_target[0], curr_astar_target[1], c='magenta', marker='s', s=80, label='Current A* WP', zorder=9)



            # all_vehicles_dict = self.engine.traffic_manager.current_traffic_data


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
        if is_emergency_brake:
            self._longitudinal_controller.reset()
            return -1.0

        speed_error = target_speed_ms - current_speed_ms
        acceleration_command = self._longitudinal_controller.step(speed_error)

        if acceleration_command > 0:
            max_accel_estim = self.control_object.config.get("max_acceleration", 5.0)
            throttle = np.clip(acceleration_command / max_accel_estim, 0.0, 1.0)
            return throttle
        else:
            max_decel_estim = self.control_object.config.get("max_deceleration", 5.0)
            brake_ratio = acceleration_command / (-max_decel_estim)
            brake = -np.clip(brake_ratio, 0.0, 1.0)
            return brake

    def _get_raw_other_vehicle_predictions(self, ego_vehicle, num_forecast_steps):
        all_vehicles_dict = self.engine.traffic_manager.current_traffic_data
        other_vehicles_objects = [
            v_obj for v_id, v_obj in all_vehicles_dict.items()
            if v_id != self.engine.traffic_manager.sdc_track_index
        ]
        adv_name = self.engine.traffic_manager.adv_name
        adv_trajs = self.engine.traffic_manager.adv_traj
        predicted_other_trajectories = []

        current_episode_step = self.engine.episode_step

        for other_v_data in other_vehicles_objects:



            # type_specific_states = self.engine.data_manager.get_scenario(self.engine.current_map.scenario_id, should_copy=False)["tracks"][obj_id]["state"]
            # vehicle_length = type_specific_states['length'][current_episode_step]
            # vehicle_width = type_specific_states['width'][current_episode_step]


            if other_v_data['type'] == 'PEDESTRIAN':
                continue

            try:
                vehicle_length = other_v_data['state']['length'][current_episode_step] if 'length' in other_v_data['state'] else ego_vehicle.LENGTH
                vehicle_width  = other_v_data['state']['width'][current_episode_step] if 'width' in other_v_data['state'] else ego_vehicle.WIDTH
            except IndexError:
                 vehicle_length = ego_vehicle.LENGTH
                 vehicle_width = ego_vehicle.WIDTH


            one_vehicle_traj = []
            if self.predict_other_v:
                if not other_v_data['state']['valid'][current_episode_step]:
                    predicted_other_trajectories.append([])
                    continue

                current_other_pos = np.array(other_v_data['state']['position'][current_episode_step,:2])
                current_other_heading = other_v_data['state']['heading'][current_episode_step]
                current_other_vel_xy = other_v_data['state']['velocity'][current_episode_step,:2]
                current_other_speed = np.linalg.norm(current_other_vel_xy)


                estimated_steer, estimated_throttle, estimated_brake = 0.0, 0.0, 0.0
                if current_episode_step > 0:
                    prev_step = current_episode_step - 1
                    if other_v_data['state']['valid'][prev_step]:
                        prev_heading = other_v_data['state']['heading'][prev_step]
                        prev_vel_xy = other_v_data['state']['velocity'][prev_step,:2]
                        prev_speed = np.linalg.norm(prev_vel_xy)

                        heading_change = wrap_to_pi(current_other_heading - prev_heading)
                        heading_rate = heading_change / self.engine.global_config["physics_world_step_size"]

                        if abs(prev_speed) > 0.1:

                            # tan(delta) = L * omega / v
                            # steer_angle = atan( L_rear_axle_to_cg * heading_rate / prev_speed)
                            # estimated_steer = steer_angle / max_steer_angle_of_model

                            estimated_steer = np.clip(heading_rate * 0.2, -1.0, 1.0)

                        accel = (current_other_speed - prev_speed) / self.engine.global_config["physics_world_step_size"]
                        if accel > 0.1:
                            estimated_throttle = np.clip(accel / 5.0, 0.0, 1.0)
                        elif accel < -0.1:
                            estimated_brake = np.clip(accel / -5.0, 0.0, 1.0)

                other_control = [estimated_steer, estimated_throttle, estimated_brake]

                temp_pos, temp_head, temp_speed = current_other_pos, current_other_heading, current_other_speed
                for _ in range(num_forecast_steps):
                    one_vehicle_traj.append((np.copy(temp_pos), temp_head, temp_speed, vehicle_width, vehicle_length))
                    next_pos, next_head, next_speed = self._bicycle_model.forward(
                        temp_pos, temp_head, temp_speed, other_control
                    )
                    temp_pos, temp_head, temp_speed = next_pos, next_head, next_speed

            else:
                if other_v_data['metadata']['object_id'] in adv_name and len(adv_trajs[adv_name.index(other_v_data['metadata']['object_id'])]):
                    adv_traj = adv_trajs[adv_name.index(other_v_data['metadata']['object_id'])]
                    for i in range(num_forecast_steps):
                        forecast_step_idx = i

                        if forecast_step_idx >= len(adv_traj):

                            if one_vehicle_traj:
                                last_valid_state = one_vehicle_traj[-1]
                                one_vehicle_traj.append(last_valid_state)
                            else:
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

                        if forecast_step_idx >= len(other_v_data['state']['position']) or\
                        not other_v_data['state']['valid'][forecast_step_idx]:

                            if one_vehicle_traj:
                                last_valid_state = one_vehicle_traj[-1]
                                one_vehicle_traj.append(last_valid_state)
                            else:
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
        ego_pos, ego_heading, ego_speed, ego_w, ego_l = ego_state_at_collision
        other_pos, other_heading, other_speed, other_w, other_l = other_state_at_collision


        relative_pos = other_pos - ego_pos

        cos_h, sin_h = math.cos(-ego_heading), math.sin(-ego_heading)
        local_other_x = relative_pos[0] * cos_h - relative_pos[1] * sin_h
        local_other_y = relative_pos[0] * sin_h + relative_pos[1] * cos_h



        ego_vel_vec = np.array([ego_speed * math.cos(ego_heading), ego_speed * math.sin(ego_heading)])

        other_vel_vec = np.array([other_speed * math.cos(other_heading), other_speed * math.sin(other_heading)])
        relative_vel_vec = other_vel_vec - ego_vel_vec


        ego_direction_vec = np.array([math.cos(ego_heading), math.sin(ego_heading)])
        relative_speed_longitudinal = np.dot(relative_vel_vec, ego_direction_vec)






        rear_threshold = -ego_l * 0.4
        front_threshold = ego_l * 0.4
        side_threshold = ego_w * 0.6 + other_w * 0.5

        is_rear = local_other_x < rear_threshold and abs(local_other_y) < side_threshold
        is_front = local_other_x > front_threshold and abs(local_other_y) < side_threshold


        is_side = not is_rear and not is_front and abs(local_other_y) < side_threshold

        if is_rear:


            return "rear", relative_speed_longitudinal
        elif is_front:

            return "front", relative_speed_longitudinal
        elif is_side:

            if local_other_y > 0:
                return "side_left", relative_speed_longitudinal
            else:
                return "side_right", relative_speed_longitudinal

        return "unknown", relative_speed_longitudinal


    def _is_path_clear_ahead(self, ego_vehicle, ego_pos_2d, ego_heading_math, ego_speed_ms,
                             predicted_other_vehicle_trajectories, check_duration_s=2.0):
        num_check_steps = int(check_duration_s / self.forecast_dt)
        ego_w = ego_vehicle.WIDTH
        ego_l = ego_vehicle.LENGTH


        temp_ego_pos = np.copy(ego_pos_2d)
        temp_ego_heading = ego_heading_math
        temp_ego_speed = ego_speed_ms

        for i in range(num_check_steps):



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


                vec_to_other = other_pred_pos - temp_ego_pos
                dist_to_other = np.linalg.norm(vec_to_other)
                if dist_to_other > ego_l * 2 + other_l:
                    continue

                angle_to_other = math.atan2(vec_to_other[1], vec_to_other[0])
                angle_diff = abs(wrap_to_pi(angle_to_other - temp_ego_heading))


                if angle_diff < math.radians(15):
                    other_obb = OBB(other_pred_pos, other_w, other_l, other_pred_heading)
                    if check_obb_intersection_sat(ego_check_obb, other_obb):
                        self.logger.info(f"INFO: Path ahead NOT clear due to object at step {i}")
                        return False
        self.logger.info("INFO: Path ahead IS clear.")
        return True

    def _predict_infractions(self, ego_vehicle, ego_pos_2d, ego_heading_math, ego_speed_ms, at_intersection):


        for lane_info in ego_vehicle.navigation.current_ref_lanes:

            if hasattr(lane_info, 'traffic_light') and lane_info.traffic_light is not None:
                traffic_light = lane_info.traffic_light
                if traffic_light.get_state() == "red":




                    dist_to_light_stop_line = norm(traffic_light.position[0] - ego_pos_2d[0],
                                                   traffic_light.position[1] - ego_pos_2d[1])
                    if dist_to_light_stop_line < 15.0:

                        vec_to_light = traffic_light.position - ego_pos_2d
                        angle_to_light = math.atan2(vec_to_light[1], vec_to_light[0])
                        if abs(wrap_to_pi(angle_to_light - ego_heading_math)) < math.pi / 4:
                            return True, "red_light"


        forecast_duration = self.forecast_duration_intersection if at_intersection else self.forecast_duration_other
        num_forecast_steps = int(forecast_duration / self.forecast_dt)
        # predicted_other_vehicle_trajectories: list of lists, each sublist is [(pos, heading, speed, width, length), ...]
        predicted_other_vehicle_trajectories = self._get_raw_other_vehicle_predictions(ego_vehicle, num_forecast_steps)


        dynamic_target_speed = self.target_speed_standard

        self.forward_vehicles_speeds = []
        if ego_vehicle.navigation and ego_vehicle.navigation.current_ref_lanes:
            ego_lane = ego_vehicle.navigation.current_ref_lanes[0]
            ego_long_curr, _ = ego_lane.local_coordinates(ego_pos_2d)

            for other_traj in predicted_other_vehicle_trajectories:
                if not other_traj: continue
                other_curr_pos = other_traj[0][0]
                other_curr_speed = other_traj[0][2]

                dist_to_other = np.linalg.norm(other_curr_pos - ego_pos_2d)
                if dist_to_other > self.dynamic_speed_perception_radius:
                    continue

                try:
                    other_long, other_lat = ego_lane.local_coordinates(other_curr_pos)

                    if (abs(other_lat) < ego_lane.width * 1.2) & (other_curr_speed > 0):
                        self.forward_vehicles_speeds.append(other_curr_speed)
                except:
                    pass

        if self.forward_vehicles_speeds:
            calculated_speed = np.percentile(self.forward_vehicles_speeds, 75)
            dynamic_target_speed = np.clip(calculated_speed,
                                           self.target_speed_standard * 0.5,
                                        #    0,
                                           self.target_speed_standard * 2.5)
            # dynamic_target_speed = min(dynamic_target_speed,max(forward_vehicles_speeds))
        else:
            dynamic_target_speed = self.target_speed_standard * 1.5




        ego_future_poses_kinematic = [] # list of (pos, heading_math, speed)
        current_ego_pos = np.copy(ego_pos_2d)
        current_ego_heading = ego_heading_math
        current_ego_speed = ego_speed_ms
        for _ in range(num_forecast_steps):

            pred_steering = self._lateral_control(current_ego_pos, current_ego_heading, current_ego_speed)

            pred_speed_error = dynamic_target_speed - current_ego_speed
            pred_accel_cmd = self._longitudinal_controller.step(pred_speed_error)
            pred_throttle = np.clip(pred_accel_cmd / 5.0, 0, 1) if pred_accel_cmd > 0 else 0

            ego_pred_control = [pred_steering, pred_throttle, 0.0]

            next_pos, next_heading, next_speed = self._bicycle_model.forward(
                current_ego_pos, current_ego_heading, current_ego_speed, ego_pred_control
            )
            ego_future_poses_kinematic.append((next_pos, next_heading, next_speed,ego_vehicle.WIDTH, ego_vehicle.LENGTH))
            current_ego_pos, current_ego_heading, current_ego_speed = next_pos, next_heading, next_speed


        imminent_collision_detected = False
        colliding_vehicle_data_at_collision = None # (pos, heading, speed, width, length)
        ego_state_at_collision = None
        collision_time_step = -1
        colliding_obj_original_idx = -1

        for i in range(num_forecast_steps):
            ego_pred_state_full = ego_future_poses_kinematic[i] # (pos, heading, speed, ego_w, ego_l)
            ego_pred_pos, ego_pred_heading, _, ego_w, ego_l = ego_pred_state_full
            ego_obb = OBB(ego_pred_pos, ego_w, ego_l, ego_pred_heading)

            for other_idx, other_v_traj in enumerate(predicted_other_vehicle_trajectories):
                if i >= len(other_v_traj) or not other_v_traj: continue

                other_pred_state_full = other_v_traj[i] # (pos, heading, speed, other_w, other_l)
                other_pred_pos, other_pred_heading, _, other_w, other_l = other_pred_state_full

                if np.linalg.norm(ego_pred_pos - other_pred_pos) > (ego_l + other_l + 5.0):
                    continue

                other_obb = OBB(other_pred_pos, other_w, other_l, other_pred_heading)

                if check_obb_intersection_sat(ego_obb, other_obb):
                    imminent_collision_detected = True
                    colliding_vehicle_data_at_collision = other_pred_state_full
                    ego_state_at_collision = ego_pred_state_full
                    collision_time_step = i
                    colliding_obj_original_idx = other_idx
                    self.logger.info(f"INFO: Collision predicted at step {i} with an object (idx: {other_idx}).")
                    break
            if imminent_collision_detected:
                break

        # visualize_trajectories(

        #         other_trajectories=predicted_other_vehicle_trajectories,
        #         colliding_vehicle_idx=3,
        #         title="[FAIL CASE] Constant Speed Prediction - No Collision Detected",
        #         dt=self.forecast_dt
        #     )



        if imminent_collision_detected:
            collision_type, rel_speed_long = self._get_collision_details(ego_state_at_collision,
                                                                         colliding_vehicle_data_at_collision,
                                                                         ego_vehicle)
            self.logger.info(f"INFO: Collision predicted type: {collision_type}, RelSpeedLong: {rel_speed_long:.2f} m/s, TimeStep: {collision_time_step}")


            if collision_type == "rear" and rel_speed_long > 0.5:







                rear_collider_current_speed = 0.0
                if colliding_obj_original_idx != -1 and predicted_other_vehicle_trajectories[colliding_obj_original_idx]:
                    rear_collider_current_speed = predicted_other_vehicle_trajectories[colliding_obj_original_idx][0][2]

                if self._is_path_clear_ahead(ego_vehicle, ego_pos_2d, ego_heading_math, ego_speed_ms, predicted_other_vehicle_trajectories):
                    self.logger.info("INFO: Rear collision predicted. Path ahead clear. Attempting to accelerate.")
                    self.local_planner_active = False
                    self.current_local_path = None

                    if rear_collider_current_speed > ego_speed_ms:

                        accelerate_target = max(rear_collider_current_speed + 2.0, self.target_speed_standard * 1.5)
                    else:
                        accelerate_target = max(ego_speed_ms + 1.0, self.target_speed_standard)




                    if self.forward_vehicles_speeds:
                        final_target_speed = max(accelerate_target,(accelerate_target+dynamic_target_speed)/2)
                    else:
                        final_target_speed = accelerate_target


                    self.logger.info(f"INFO: Rear Evade. EgoSpd:{ego_speed_ms:.1f}, RearColliderSpd:{rear_collider_current_speed:.1f}, AccelTarget:{accelerate_target:.1f}, DynTarget:{dynamic_target_speed:.1f}, FinalTarget:{final_target_speed:.1f}")

                    return True, "rear_collision_accelerate", colliding_obj_original_idx, final_target_speed
                else:
                    self.logger.info("WARN: Rear collision predicted, but path ahead is NOT clear. Falling back to A* or emergency.")
                    return True, "rear_collision_accelerate", colliding_obj_original_idx, rear_collider_current_speed + 1.0

            else:
                return True, "collision", colliding_obj_original_idx, self.target_speed_halt


            # if not self.local_planner_active:
            #     self.logger.info("INFO: Imminent collision (not rear-accelerate type or front blocked). Attempting A* local planning...")


            #     if not ego_vehicle.navigation or not ego_vehicle.navigation.current_ref_lanes:
            #         self.logger.info("WARN: A* planning: No navigation path available for goal setting.")
            #         return True, "collision_emergency", colliding_obj_original_idx, self.target_speed_halt

            #     current_ref_lane = ego_vehicle.navigation.current_ref_lanes[0]

            #     goal_pos_astar = current_ref_lane.position(current_long + self.a_star_goal_distance, 0)

            #     planned_path = self.a_star_planner.plan_path(

            #         goal_pos_astar,
            #         predicted_other_vehicle_trajectories,
            #         current_ref_lane
            #     )
            #     if planned_path and len(planned_path) > 1:
            #         self.logger.info(f"INFO: A* planner found a path with {len(planned_path)} waypoints.")
            #         self.current_local_path = planned_path
            #         self.current_local_path_index = 0
            #         self.local_planner_active = True


            #         return False, "collision_astar_planned", None, dynamic_target_speed
            #     else:
            #         self.logger.info("WARN: A* planner failed to find a safe path. Emergency braking.")
            #         return True, "collision_emergency", colliding_obj_original_idx, self.target_speed_halt



            #     self.logger.info(f"WARN: Collision predicted WHILE local A* path is active. Re-evaluating. Type: {collision_type}")

            #     if collision_type == "rear" and rel_speed_long > 0.5 and \
            #        self._is_path_clear_ahead(ego_vehicle, ego_pos_2d, ego_heading_math, ego_speed_ms, predicted_other_vehicle_trajectories):
            #         self.logger.info("INFO: Rear collision during A*. Path ahead clear. Cancelling A* and accelerating.")
            #         self.local_planner_active = False
            #         self.current_local_path = None
            #         rear_collider_current_speed = predicted_other_vehicle_trajectories[colliding_obj_original_idx][0][2]

            #         if rear_collider_current_speed > ego_speed_ms:
            #             accelerate_target = max(rear_collider_current_speed + 2.0, self.target_speed_standard * 1.5)
            #         else:
            #             accelerate_target = max(ego_speed_ms + 1.0, self.target_speed_standard)
            #         # final_target_speed = min(accelerate_target, dynamic_target_speed if forward_vehicles_speeds else accelerate_target*1.2)
            #         final_target_speed = max(accelerate_target, ego_speed_ms)
            #         self.logger.info(f"INFO: Rear Evade (during A*). EgoSpd:{ego_speed_ms:.1f}, RearColliderSpd:{rear_collider_current_speed:.1f}, AccelTarget:{accelerate_target:.1f}, DynTarget:{dynamic_target_speed:.1f}, FinalTarget:{final_target_speed:.1f}")
            #         return True, "rear_collision_accelerate", colliding_obj_original_idx, final_target_speed

            #         self.logger.info("WARN: A* path unsafe, not simple rear-accelerate. Attempting A* replan or emergency brake.")

            #         self.current_local_path = None


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

            #             self.logger.info("A* planner: No path found or max steps reached.")
            #             return True, "collision_emergency", colliding_obj_original_idx, self.target_speed_halt

        return False, None, None, dynamic_target_speed

    def _lateral_control_for_prediction(self, ego_pos_2d, ego_heading, ego_speed_ms, ego_vehicle_obj):
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
