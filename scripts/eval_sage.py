import argparse
import numpy as np
import os
import sys
import logging
from tqdm import tqdm
import time
import pygame
import pandas as pd
import torch
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# MetaDrive Imports
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from saferl_algo import TD3
# AdvGen Imports
from advgen.modeling.vectornet import VectorNet
from advgen.adv_utils import process_data
import advgen.utils as advgen_utils
from advgen.adv_generator import AdvGenerator as OriginalAdvGenerator  # For internal data parsing
from scipy.stats import wasserstein_distance
from sage.metrics import get_kinematic_profiles
from sage.splits import scenario_ids

def moving_average(data, window_size):
    """Smooth a numeric sequence with an edge-padded moving average."""
    interval = np.pad(data, window_size // 2, 'edge')
    window = np.ones(int(window_size)) / float(window_size)
    res = np.convolve(interval, window, 'valid')
    return res


def get_polyline_yaw(polyline):
    """Compute yaw for each trajectory point."""
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


def get_polyline_vel(polyline):
    """Compute velocity from displacement with a 0.1 second time step."""
    polyline_post = np.roll(polyline, shift=-1, axis=0)
    polyline_post[-1] = polyline[-1]
    diff = polyline_post - polyline
    polyline_vel = diff / 0.1
    return polyline_vel


def Intersect(l1, l2):
    """Return whether two line segments intersect."""
    v1 = (l1[0] - l2[0], l1[1] - l2[1])
    v2 = (l1[0] - l2[2], l1[1] - l2[3])
    v0 = (l1[0] - l1[2], l1[1] - l1[3])
    a = v0[0] * v1[1] - v0[1] * v1[0]
    b = v0[0] * v2[1] - v0[1] * v2[0]
    if a * b >= 0: return False

    temp = l1;
    l1 = l2;
    l2 = temp
    v1 = (l1[0] - l2[0], l1[1] - l2[1])
    v2 = (l1[0] - l2[2], l1[1] - l2[3])
    v0 = (l1[0] - l1[2], l1[1] - l1[3])
    c = v0[0] * v1[1] - v0[1] * v1[0]
    d = v0[0] * v2[1] - v0[1] * v2[0]
    return c * d < 0


from sage.rewards import calculate_map_violation_penalty, calculate_realism_penalty, calculate_adversarial_reward

class MotionModel:
    """DenseTNT wrapper used by SAGE generation."""

    def __init__(self, model_path: str, device: torch.device):
        parser = argparse.ArgumentParser()
        advgen_utils.add_argument(parser)
        parser.set_defaults(
            other_params=['l1_loss', 'densetnt', 'goals_2D', 'enhance_global_graph', 'laneGCN', 'point_sub_graph',
                          'laneGCN-4', 'stride_10_2', 'raster', 'train_pair_interest'])
        parser.set_defaults(mode_num=32, future_frame_num=80)
        args, _ = parser.parse_known_args()

        dummy_logger = logging.getLogger(f"dummy_logger_{os.path.basename(model_path)}")
        logging.basicConfig(level=logging.WARNING)
        advgen_utils.init(args, dummy_logger)

        self.args = args
        self.device = device
        self.model = VectorNet(args).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def _get_full_context(self, batch_data: list):
        """Compute shared context states for all agents in the scene."""
        all_mappings = batch_data[0]  # [mapping_ego, mapping_adv]
        element_states_batch_full, _ = self.model.forward_encode_sub_graph(
            all_mappings, [m['matrix'] for m in all_mappings], [m['polyline_spans'] for m in all_mappings],
            self.device, len(all_mappings)
        )
        merged_inputs_full, inputs_lengths_full = advgen_utils.merge_tensors(element_states_batch_full,
                                                                             device=self.device)

        batch_size_effective = len(all_mappings)
        max_poly_num_full = merged_inputs_full.shape[1]
        attention_mask_full = torch.zeros([batch_size_effective, max_poly_num_full, max_poly_num_full],
                                          device=self.device)
        for i, length in enumerate(inputs_lengths_full):
            attention_mask_full[i, :length, :length].fill_(1)

        hidden_states_full = self.model.global_graph(merged_inputs_full, attention_mask_full, all_mappings)
        return all_mappings, merged_inputs_full, hidden_states_full, inputs_lengths_full

    @torch.no_grad()
    def get_goal_scores(self, batch_data: list) -> torch.Tensor:
        """Return candidate goal log-probabilities for the adversarial agent."""
        all_mappings, merged_inputs_full, hidden_states_full, inputs_lengths_full = self._get_full_context(batch_data)

        if len(all_mappings) < 2:
            raise ValueError("SAGE requires at least two agents (ego and adversarial agent).")

        adv_mapping = all_mappings[1]
        goals_2D_tensor_adv = torch.tensor(adv_mapping['goals_2D'], device=self.device, dtype=torch.float)

        scores = self.model.decoder.get_scores(
            goals_2D_tensor_adv,
            merged_inputs_full, hidden_states_full, inputs_lengths_full,
            i=1,
            mapping=all_mappings, device=self.device
        )
        return scores

    @torch.no_grad()
    def generate_trajectories_for_goals(self, batch_data: list, top_k_goals: np.ndarray) -> np.ndarray:
        """Generate trajectories for selected target goals."""
        all_mappings, merged_inputs_full, hidden_states_full, inputs_lengths_full = self._get_full_context(batch_data)

        if len(all_mappings) < 2:
            raise ValueError("SAGE requires at least two agents (ego and adversarial agent).")
        adv_mapping = all_mappings[1]

        k = len(top_k_goals)
        goals_2D_tensor_topk = torch.tensor(top_k_goals, device=self.device, dtype=torch.float)
        targets_feature_topk = self.model.decoder.goals_2D_mlps(goals_2D_tensor_topk)

        hidden_attention_topk = self.model.decoder.tnt_cross_attention(
            targets_feature_topk.unsqueeze(0),
            merged_inputs_full[1][:inputs_lengths_full[1]].unsqueeze(0)
        ).squeeze(0)

        predict_trajs_tensor_local = self.model.decoder.tnt_decoder(
            torch.cat([hidden_states_full[1, 0, :].unsqueeze(0).expand(k, -1),
                       targets_feature_topk,
                       hidden_attention_topk], dim=-1)
        ).view([k, self.model.decoder.future_frame_num, 2])

        normalizer = adv_mapping['normalizer']
        predict_trajs_np_local = predict_trajs_tensor_local.cpu().numpy()
        predict_trajs_np_world = np.array([normalizer(traj, reverse=True) for traj in predict_trajs_np_local])
        return predict_trajs_np_world


from sage.model_soup import create_souped_model  # noqa: E402


class SAGEAdvGeneratorForEval:
    def __init__(self, adv_model_path, real_model_path, sage_args):
        self.sage_args = sage_args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading SAGE models...")
        self.adversarial_model = MotionModel(adv_model_path, self.device)
        self.realism_model = MotionModel(real_model_path, self.device)
        print(f"Using device: {self.device}")

        _parser = argparse.ArgumentParser()
        _parser.add_argument('--OV_traj_num', type=int, default=32)
        _parser.add_argument('--AV_traj_num', type=int, default=1)
        self._internal_adv_generator = OriginalAdvGenerator(_parser)
        self.adv_traj = None
        self.env = None

        self.args = self.adversarial_model.args

    def before_episode(self, env):
        self.env = env
        self._internal_adv_generator.before_episode(env)

    def log_AV_history(self):
        """Record ego-vehicle history for closed-loop generation."""
        self._internal_adv_generator.log_AV_history()

    def after_episode(self, update_AV_traj=False, mode='train',n=10):
        """Update the stored ego-vehicle trajectory after an episode."""
        self._internal_adv_generator.after_episode(update_AV_traj=update_AV_traj, mode=mode,n=n)

    @property
    def adv_agent(self):
        return self.storage[self.env.current_seed].get('adv_agent')

    @property
    def storage(self):
        return self._internal_adv_generator.storage

    # @property
    # def env(self):
    #     return self._internal_adv_generator.env

    def get_data_for_scenario(self, env, mode='train'):
        """Collect model inputs, scene metadata, and the ego trajectory."""
        self._internal_adv_generator.before_episode(env)
        traffic_motion_feat = self.storage[env.current_seed]['traffic_motion_feat']
        batch_data = process_data(traffic_motion_feat, self.args)
        adv_info = self.storage[env.current_seed]['adv_info']
        ego_info = self.storage[env.current_seed]['ego_info']
        # ego_gt_future_traj_x = traffic_motion_feat['state/future/x'].numpy()[0, :, np.newaxis]
        # ego_gt_future_traj_y = traffic_motion_feat['state/future/y'].numpy()[0, :, np.newaxis]
        # ego_gt_future_traj = np.concatenate([ego_gt_future_traj_x, ego_gt_future_traj_y], axis=-1)
        adv_past_traj = self.storage[env.current_seed]['adv_past']
        raw_map_features = env.engine.data_manager.get_scenario(env.current_seed)['map_features']

        ego_traj_deque = self._internal_adv_generator.storage[env.current_seed].get('AV_trajs_eval' if mode == 'eval' else 'AV_trajs')

        if ego_traj_deque and len(ego_traj_deque) > 0:
            latest_ego_traj = ego_traj_deque[-1]
            print(f"[{mode.upper()} mode] Using latest RL agent trajectory for generation.")
        else:
            print(f"[{mode.upper()} mode] RL agent trajectory not available, falling back to GT trajectory.")
            ego_gt_future_traj_x = traffic_motion_feat['state/future/x'].numpy()[0, :, np.newaxis]
            ego_gt_future_traj_y = traffic_motion_feat['state/future/y'].numpy()[0, :, np.newaxis]
            latest_ego_traj = np.concatenate([ego_gt_future_traj_x, ego_gt_future_traj_y], axis=-1)

        return batch_data, traffic_motion_feat, adv_info, ego_info, latest_ego_traj, adv_past_traj, raw_map_features

    def generate_with_sage(self):
        """Generate a SAGE adversarial trajectory and return metrics."""
        batch_data, traffic_motion_feat, adv_info, ego_info, ego_agent_future_traj, adv_past_traj, raw_map_features = self.get_data_for_scenario(
            self.env)

        w_adv = self.sage_args.w_adv
        w_real = 1.0 - w_adv

        souped_vectornet_model = create_souped_model(
            self.adversarial_model.model,
            self.realism_model.model,
            w_adv,
            w_real,
            self.device)

        temp_souped_model_wrapper = deepcopy(self.adversarial_model)
        temp_souped_model_wrapper.model = souped_vectornet_model

        pred_trajs_list, _, _ = temp_souped_model_wrapper.model(batch_data[0], self.device, return_tensors_for_hgpo=True)
        adv_candidate_trajs_np = pred_trajs_list[1]  # Shape: [32, 80, 2]
        rewards = []
        for traj in adv_candidate_trajs_np:
            realism_penalties = calculate_realism_penalty(traj, adv_info)
            total_realism_penalty = (realism_penalties["behavior_penalty"] + realism_penalties["kinematic_penalty"])
            adversarial_rew, is_collision = calculate_adversarial_reward(traj, ego_agent_future_traj, adv_info,
                                                                         ego_info)
            map_violations = calculate_map_violation_penalty(traj, raw_map_features, traffic_motion_feat, adv_info)
            total_map_penalty = sum(map_violations.values())
            total_reward = w_adv * adversarial_rew - w_real * total_realism_penalty - total_map_penalty
            rewards.append(total_reward)

        winner_idx = np.argmax(rewards)
        final_traj_points = adv_candidate_trajs_np[winner_idx]
        map_violations = calculate_map_violation_penalty(final_traj_points, raw_map_features, traffic_motion_feat,
                                                         adv_info)
        realism_penalties = calculate_realism_penalty(final_traj_points, adv_info)
        adversarial_rew, is_collision = calculate_adversarial_reward(final_traj_points, ego_agent_future_traj, adv_info,
                                                                     ego_info)

        adv_gt_x = traffic_motion_feat['state/future/x'].numpy()[1, :, np.newaxis]
        adv_gt_y = traffic_motion_feat['state/future/y'].numpy()[1, :, np.newaxis]
        adv_gt_valid = traffic_motion_feat['state/future/valid'].numpy()[1, :]

        adv_gt_traj_full = np.concatenate([adv_gt_x, adv_gt_y], axis=-1)
        adv_gt_traj = adv_gt_traj_full[adv_gt_valid]

        current_gen_profiles = get_kinematic_profiles(final_traj_points)
        current_gt_profiles = get_kinematic_profiles(adv_gt_traj)
        # ==============================================================================

        metrics = {
            "collision": 1.0 if is_collision else 0.0,
            "adversarial_reward": adversarial_rew,
            **map_violations,
            **realism_penalties,
        }

        adv_pos = np.concatenate((adv_past_traj, final_traj_points), axis=0)
        adv_yaw = get_polyline_yaw(adv_pos).reshape(-1, 1)
        adv_vel = get_polyline_vel(adv_pos)
        self.adv_traj = list(np.concatenate((adv_pos, adv_vel, adv_yaw), axis=1))

        return self.adv_traj, metrics, current_gen_profiles, current_gt_profiles, traffic_motion_feat

def evaluate_metrics(env,adv_generator,adv_traj_from_gen,raw_map_features, traffic_motion_feat):
    storage_entry = adv_generator._internal_adv_generator.storage[env.current_seed]
    adv_info = storage_entry['adv_info']
    ego_info = storage_entry['ego_info']

    ego_traj_deque = storage_entry.get('AV_trajs')
    if ego_traj_deque and len(ego_traj_deque) > 0:
        latest_ego_traj = ego_traj_deque[-1]
    else:
        raise ValueError
    adv_traj_np = np.array(adv_traj_from_gen)[11:, :2]  # only the trajectory

    map_violations = calculate_map_violation_penalty(adv_traj_np, raw_map_features, traffic_motion_feat, adv_info)
    realism_penalties = calculate_realism_penalty(adv_traj_np, adv_info)
    adversarial_rew, is_collision = calculate_adversarial_reward(adv_traj_np, latest_ego_traj, adv_info, ego_info)

    adv_gt_x = traffic_motion_feat['state/future/x'].numpy()[1, :, np.newaxis]
    adv_gt_y = traffic_motion_feat['state/future/y'].numpy()[1, :, np.newaxis]
    adv_gt_valid = traffic_motion_feat['state/future/valid'].numpy()[1, :]
    adv_gt_traj_full = np.concatenate([adv_gt_x, adv_gt_y], axis=-1)
    adv_gt_traj = adv_gt_traj_full[adv_gt_valid]


    current_metrics = {
        "collision": 1.0 if is_collision else 0.0,
        "adversarial_reward": adversarial_rew,
        **map_violations,
        **realism_penalties,
    }
    return current_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # SAGE specific arguments
    parser.add_argument('--adv_model_path', type=str, default='./advgen/finetuned/hgpo_finetuned_model_adv_best.bin',
                        help="Path to the 'adversarial' fine-tuned model.")
    parser.add_argument('--real_model_path', type=str, default='./advgen/finetuned/hgpo_finetuned_model_real_best.bin',
                        help="Path to the 'realism' fine-tuned model.")
    parser.add_argument('--w_adv', type=float, default=0.5,
                        help="Weight for the adversarial objective in SAGE. w_real = 1 - w_adv.")

    # Standard environment arguments
    parser.add_argument('--data_directory', type=str, default='./raw_scenes_500')
    parser.add_argument('--split_file', type=str, default='./configs/splits/sage_womd_500.json')
    parser.add_argument('--split', type=str, default='eval')
    parser.add_argument('--max_scenarios', type=int, default=None,
                        help="Optional cap for quick smoke tests.")
    parser.add_argument('--use_render', action='store_true',
                        help="Render and save top-down frames. Disabled by default.")
    parser.add_argument('--results_path', type=str, default='./saved_results/sage_eval_results.csv',
                        help="CSV path for per-scenario metrics.")
    parser.add_argument('--td3_policy_path', type=str, default=None,
                        help="Optional TD3 policy path for RL-policy evaluation.")
    args = parser.parse_args()

    adv_generator = SAGEAdvGeneratorForEval(
        adv_model_path=args.adv_model_path,
        real_model_path=args.real_model_path,
        sage_args=args
    )

    extra_args = dict(mode="top_down", film_size=(2200, 2200), screen_size=(600, 600))

    time_cost = 0.
    results_list = []
    eval_ids = scenario_ids(args.split_file, split=args.split, max_scenarios=args.max_scenarios)


    test_rl = False

    if test_rl:
        config_test = dict(
            data_directory=args.data_directory,
            start_scenario_index = 0,
            num_scenarios=500,
            # crash_vehicle_done=True,
            sequential_seed = True,
            force_reuse_object_name = True,
            horizon = 50,
            no_light = True,
            no_static_vehicles = True,
            reactive_traffic = False,
            vehicle_config=dict(
            lidar = dict(num_lasers=30,distance=50, num_others=3),
            side_detector = dict(num_lasers=30),
            lane_line_detector = dict(num_lasers=12)),
        )
        env = WaymoEnv(config=config_test)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0] 
        max_action = float(env.action_space.high[0])
        kwargs = {
        "state_dim": state_dim,  # 101
        "action_dim": action_dim,  # 2
        "max_action": max_action,  # 1
        "discount": 0.99,
        "tau": 0.005,
        }
        kwargs["policy_noise"] = 0.2 * max_action
        kwargs["noise_clip"] = 0.5 * max_action
        kwargs["policy_freq"] = 2
        policy = TD3.TD3(**kwargs)
        if args.td3_policy_path is None:
            raise ValueError("--td3_policy_path is required when test_rl is enabled.")
        policy.load(args.td3_policy_path)

    else:
        env = WaymoEnv({
            "agent_policy": ReplayEgoCarPolicy,  
            "reactive_traffic": False,
            "use_render": False,
            "data_directory": args.data_directory,
            "num_scenarios": max(eval_ids) + 1 if eval_ids else 0,
            "force_reuse_object_name": True,
            "sequential_seed": True,
            "start_scenario_index": 0,
        })

    pbar = tqdm(eval_ids, desc=f"SAGE Adversarial Evaluation (w_adv={args.w_adv})")
    attack_cnt = 0
    render = args.use_render
    image_name = f'sage_w_{args.w_adv}_replay'

    all_gen_profiles = {"vel": [], "acc": [], "yaw_rate": []}
    all_gt_profiles = {"vel": [], "acc": [], "yaw_rate": []}
    for i in pbar:
        try:
            o = env.reset(force_seed=i)
            env.vehicle.ego_crash_flag = False
        except Exception as e:
            print(f"Error resetting env for seed {i}: {e}. Skipping.")
            continue

        done = False
        adv_generator.before_episode(env)
        if render:
            ret = env.render(**extra_args)
            if not os.path.exists("image/{}/top_down_{}/".format(image_name, env.current_seed)):
                os.makedirs("image/{}/top_down_{}/".format(image_name, env.current_seed))  
            pygame.image.save(ret, "image/{}/top_down_{}/episode_{}.png".format(image_name, env.current_seed, env.episode_step))
        if adv_generator.adv_agent and render:
            env.engine._top_down_renderer.set_adv(adv_generator.adv_agent)

        while not done:
            adv_generator.log_AV_history()
            if test_rl:
                action = policy.select_action(np.array(o)).clip(-max_action, max_action)
                o, r, done, info = env.step(action)
            else:
                o, r, done, info = env.step([1.0, 0.])
            if render:
                ret = env.render(**extra_args, text={'Replay': 'Raw Scenario'})
                pygame.image.save(ret, "image/{}/top_down_{}/episode_{}.png".format(image_name, env.current_seed, env.episode_step))

            # crash = env.vehicle.ego_crash_flag
            # if crash:
            #     break
            if env.episode_step > 91:
                adv_generator.log_AV_history()
                break
            

        adv_generator.after_episode()

        o = env.reset(force_seed=i)
        env.vehicle.ego_crash_flag = False
        done = False

        t0 = time.time()
        # before_episode is called inside get_data_for_scenario, so we just call the main function
        # adv_generator.before_episode(env) # This is now handled internally

        sage_traj_for_sim, current_metrics, gen_profiles, gt_profiles, traffic_motion_feat = adv_generator.generate_with_sage()

        t1 = time.time()
        time_cost += t1 - t0

        if sage_traj_for_sim and current_metrics:
            # for key in all_gen_profiles.keys():
            #     all_gen_profiles[key].extend(gen_profiles[key])
            #     all_gt_profiles[key].extend(gt_profiles[key])
            # results_list.append(current_metrics)
            env.engine.traffic_manager.set_adv_info(adv_generator.adv_agent, sage_traj_for_sim)
            sage_adv_traj_for_sim = sage_traj_for_sim.copy()
        else:
            print(f"Skipping simulation for seed {i} due to generation failure.")
            adv_generator.after_episode()
            continue

        crash = []

        while not done:
            adv_generator.log_AV_history()
            if test_rl:
                action = policy.select_action(np.array(o)).clip(-max_action, max_action)
                o, r, done, info = env.step(action)
            else:
                o, r, done, info = env.step([1.0, 0.])
            if render:
                ret = env.render(**extra_args, text={'SAGE Generate': f'Safety-Critical Scenario, seed {i}'})
                pygame.image.save(ret, "image/{}/top_down_{}/episode_adv_{}.png".format(image_name,env.current_seed, env.episode_step))
                
            crash.append(env.vehicle.ego_crash_flag)
            
            if env.vehicle.ego_crash_flag:
                adv_generator.log_AV_history()
                break
            if done:
                adv_generator.log_AV_history()
                break
            if env.episode_step > 91:
                adv_generator.log_AV_history()
                break
        if max(crash) == True:
            attack_cnt += 1
        if len(adv_generator._internal_adv_generator.ego_traj) <= 11:
            continue
        adv_generator.after_episode(update_AV_traj=True,n=1)
        raw_map_features = env.engine.data_manager.get_scenario(env.current_seed)['map_features']
        current_metrics = evaluate_metrics(env,adv_generator,sage_adv_traj_for_sim,raw_map_features,traffic_motion_feat)
        for key in all_gen_profiles.keys():
            all_gen_profiles[key].extend(gen_profiles[key])
            all_gt_profiles[key].extend(gt_profiles[key])
        results_list.append(current_metrics)


        if results_list:
            df = pd.DataFrame(results_list)
            avg_metrics = df.mean()

            realism_keys = [k for k in avg_metrics.index if 'penalty' in k and 'kinematic' in k or 'behavior' in k]
            map_keys = [k for k in avg_metrics.index if 'map' in k or 'solid_line' in k or 'object' in k]
            total_real_pen = avg_metrics[realism_keys].sum()
            total_map_pen = avg_metrics[map_keys].sum()

            pbar.set_postfix({
                "CollRate_env": f"{attack_cnt / len(results_list):.2%}",
                "CollRate": f"{avg_metrics.get('collision', 0):.2%}",
                "AvgAdvRew": f"{avg_metrics.get('adversarial_reward', 0):.2f}",
                "AvgRealPen": f"{total_real_pen:.2f}",
                "AvgMapPen": f"{total_map_pen:.2f}",
                "AvgTime": f"{time_cost / len(results_list):.2f}s"
            })

    env.close()

    final_dist_metrics = {}
    if results_list:
        for key in all_gen_profiles.keys():
            all_gen_profiles[key] = np.array(all_gen_profiles[key])
            all_gt_profiles[key] = np.array(all_gt_profiles[key])

        wd_vel = wasserstein_distance(all_gen_profiles["vel"], all_gt_profiles["vel"])
        wd_acc = wasserstein_distance(all_gen_profiles["acc"], all_gt_profiles["acc"])
        wd_yaw_rate = wasserstein_distance(all_gen_profiles["yaw_rate"], all_gt_profiles["yaw_rate"])

        realism_meta_metric = (wd_vel + wd_acc + wd_yaw_rate) / 3.0

        final_dist_metrics = {
            "wd_velocity": wd_vel,
            "wd_acceleration": wd_acc,
            "wd_yaw_rate": wd_yaw_rate,
            "realism_meta_metric": realism_meta_metric
        }
    print("\n" + "=" * 70)
    print(" " * 20 + f"SAGE Model Evaluation Summary (w_adv={args.w_adv})")
    print("=" * 70)

    if not results_list:
        print("No scenarios were successfully evaluated.")
    else:
        results_df = pd.DataFrame(results_list)
        os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
        results_df.to_csv(args.results_path, index=False)
        print(f"Per-scenario metrics saved to: {args.results_path}")
        summary_stats = results_df.mean()
        for key, value in final_dist_metrics.items():
            summary_stats[key] = value
        num_scenarios = len(results_df)

        print(f"Total Scenarios Tested: {num_scenarios}")
        print(f"Average Generation Time: {time_cost / num_scenarios:.4f} s/scenario")
        print("-" * 70)

        print(f"{'Overall & Adversarial Metrics':<45} | {'Value':<20}")
        print("-" * 70)
        print(f"{'Collision Rate':<45} | {summary_stats.get('collision', 0):.2%}")
        print(f"{'Adversarial Reward':<45} | {summary_stats.get('adversarial_reward', 0):.4f}")
        print("-" * 70)

        print(f"{'Realism Penalties':<45} | {'Value':<20}")
        print("-" * 70)
        realism_keys = sorted(
            [k for k in summary_stats.index if 'penalty' in k and 'kinematic' in k or 'behavior' in k])
        for key in realism_keys:
            print(f"{key:<45} | {summary_stats.get(key, 0):.4f}")
        print("-" * 70)

        print(f"{'Distributional Realism Metrics (WD)':<45} | {'Value':<20}")
        print("-" * 70)
        dist_keys = sorted([
            'wd_velocity', 'wd_acceleration', 'wd_yaw_rate', 'realism_meta_metric'
        ])
        for key in dist_keys:
            print(f"{key:<45} | {summary_stats.get(key, 0):.4f}")
        print("-" * 70)
        print(f"{'Map Compliance Penalties':<45} | {'Value':<20}")
        print("-" * 70)
        map_keys = sorted([k for k in summary_stats.index if 'map' in k or 'solid_line' in k or 'object' in k])
        if not map_keys:
            print(f"{'No map violation metrics found.':<45} | {'N/A':<20}")
        else:
            for key in map_keys:
                print(f"{key:<45} | {summary_stats.get(key, 0):.4f}")

    print("=" * 70)
