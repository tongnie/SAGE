import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

# MetaDrive Imports
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

from advgen.adv_utils import process_data
from advgen.modeling.vectornet import VectorNet
import advgen.utils as advgen_utils
from advgen.adv_generator import AdvGenerator as OriginalAdvGenerator
from torch.utils.tensorboard import SummaryWriter

# RL Algorithm Imports
from saferl_algo import TD3, utils
from saferl_plotter.logger import SafeLogger


def moving_average(data, window_size):
    """Smooth a numeric sequence with an edge-padded moving average."""
    interval = np.pad(data, window_size // 2, 'edge')
    window = np.ones(int(window_size)) / float(window_size)
    res = np.convolve(interval, window, 'valid')
    return res


def get_polyline_yaw(polyline):
    """Compute yaw for each trajectory point."""
    if len(polyline) < 2: return np.zeros(len(polyline))
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
    v1 = (l1[0] - l2[0], l1[1] - l2[1]);
    v2 = (l1[0] - l2[2], l1[1] - l2[3]);
    v0 = (l1[0] - l1[2], l1[1] - l1[3])
    a = v0[0] * v1[1] - v0[1] * v1[0];
    b = v0[0] * v2[1] - v0[1] * v2[0]
    temp = l1;
    l1 = l2;
    l2 = temp
    v1 = (l1[0] - l2[0], l1[1] - l2[1]);
    v2 = (l1[0] - l2[2], l1[1] - l2[3]);
    v0 = (l1[0] - l1[2], l1[1] - l1[3])
    c = v0[0] * v1[1] - v0[1] * v1[0];
    d = v0[0] * v2[1] - v0[1] * v2[0]
    return a * b < 0 and c * d < 0


class MotionModel:
    def __init__(self, model_path: str, device: torch.device):
        parser = argparse.ArgumentParser()
        advgen_utils.add_argument(parser)
        parser.set_defaults(
            other_params=['l1_loss', 'densetnt', 'goals_2D', 'enhance_global_graph', 'laneGCN', 'point_sub_graph',
                          'laneGCN-4', 'stride_10_2', 'raster', 'train_pair_interest'])
        parser.set_defaults(mode_num=32, future_frame_num=80)
        args, _ = parser.parse_known_args()

        dummy_logger = logging.getLogger(f"dummy_logger_{model_path}")
        advgen_utils.init(args, dummy_logger)

        self.args = args
        self.device = device
        self.model = VectorNet(args).to(self.device)
        print(f"Loading motion model from: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def _get_full_context_for_batch(self, batch_data: list):
        """Compute shared context states for all agents in the batch."""
        all_mappings = batch_data[0]
        element_states_batch, _ = self.model.forward_encode_sub_graph(
            all_mappings, [m['matrix'] for m in all_mappings], [m['polyline_spans'] for m in all_mappings],
            self.device, len(all_mappings))

        merged_inputs, inputs_lengths = advgen_utils.merge_tensors(element_states_batch, device=self.device)

        batch_size = len(all_mappings)
        max_poly_num = merged_inputs.shape[1]
        attention_mask = torch.zeros([batch_size, max_poly_num, max_poly_num], device=self.device)
        for i, length in enumerate(inputs_lengths):
            attention_mask[i, :length, :length].fill_(1)

        hidden_states = self.model.global_graph(merged_inputs, attention_mask, all_mappings)
        return merged_inputs, hidden_states, inputs_lengths, all_mappings

    @torch.no_grad()
    def get_goal_scores(self, batch_data: list) -> torch.Tensor:
        """Return candidate goal log-probabilities for the adversarial agent."""
        merged_inputs, hidden_states, inputs_lengths, all_mappings = self._get_full_context_for_batch(batch_data)

        adv_agent_idx = 1
        adv_mapping = all_mappings[adv_agent_idx]
        goals_2D_tensor = torch.tensor(adv_mapping['goals_2D'], device=self.device, dtype=torch.float)

        scores = self.model.decoder.get_scores(
            goals_2D_tensor, merged_inputs, hidden_states, inputs_lengths,
            adv_agent_idx, all_mappings, self.device
        )
        return scores

    @torch.no_grad()
    def generate_trajectory_from_goal(self, batch_data: list, goal_pos: np.ndarray) -> np.ndarray:
        """Generate a trajectory for one selected target goal."""
        merged_inputs, hidden_states, inputs_lengths, all_mappings = self._get_full_context_for_batch(batch_data)

        adv_agent_idx = 1
        adv_mapping = all_mappings[adv_agent_idx]

        # goal_pos.shape: (2,)
        # goal_tensor.shape: [1, 2]  (batch_size=1, feature_dim=2)
        goal_tensor = torch.tensor(goal_pos, device=self.device, dtype=torch.float).unsqueeze(0)

        # target_feature.shape: [1, hidden_dim]
        target_feature = self.model.decoder.goals_2D_mlps(goal_tensor)

        # `target_feature_unsqueezed`.shape: [1, 1, hidden_dim]
        target_feature_unsqueezed = target_feature.unsqueeze(1)

        hidden_attention = self.model.decoder.tnt_cross_attention(
            target_feature_unsqueezed,
            merged_inputs[adv_agent_idx][:inputs_lengths[adv_agent_idx]].unsqueeze(0)
        )

        # `hidden_attention`.shape: [1, 1, hidden_dim] -> .squeeze(1) -> [1, hidden_dim]
        hidden_attention_squeezed = hidden_attention.squeeze(1)

        predict_traj_local = self.model.decoder.tnt_decoder(
            torch.cat([hidden_states[adv_agent_idx, 0, :].unsqueeze(0),
                       target_feature,
                       hidden_attention_squeezed], dim=-1)
        ).view([self.model.decoder.future_frame_num, 2])

        predict_traj_world = adv_mapping['normalizer'](predict_traj_local.cpu().numpy(), reverse=True)
        return predict_traj_world


from sage.model_soup import create_souped_model
from copy import deepcopy
from sage.rewards import calculate_map_violation_penalty, calculate_realism_penalty, calculate_adversarial_reward


class SAGEAdvGeneratorForRL:
    def __init__(self, parser, sage_args):
        self.sage_args = sage_args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.adversarial_model = MotionModel(sage_args.adversarial_model_path, self.device)
        self.realism_model = MotionModel(sage_args.realism_model_path, self.device)
        print("SAGE models (adversarial and realism experts) loaded successfully.")

        self.current_w_adv = sage_args.initial_w_adv

        _parser = argparse.ArgumentParser()
        _parser.add_argument('--OV_traj_num', type=int, default=32)
        _parser.add_argument('--AV_traj_num', type=int, default=5)
        _args, _ = _parser.parse_known_args()

        advgen_parser_for_original = argparse.ArgumentParser()
        advgen_parser_for_original.add_argument('--AV_traj_num', type=int, default=_args.AV_traj_num)

        self._internal_adv_generator = OriginalAdvGenerator(advgen_parser_for_original)

        self.adv_traj = None
        self.env = None

    def update_weights(self, global_timestep: int= None, max_timesteps: int= None):
        """Update the curriculum adversarial weight."""
        if global_timestep is None:
            global_timestep = 0
        if max_timesteps is None:
            max_timesteps = self.sage_args.max_timesteps
        progress_cap = max_timesteps * 2 / 3
        progress = min(global_timestep / progress_cap, 1.0)
        self.current_w_adv = self.sage_args.initial_w_adv + \
                             (self.sage_args.final_w_adv - self.sage_args.initial_w_adv) * progress


    def set_evaluation_weights(self, w_adv: float):
        """Set a fixed evaluation-time adversarial weight."""
        self._training_w_adv = self.current_w_adv
        self.current_w_adv = w_adv
        print(f"Set evaluation mode with fixed w_adv = {self.current_w_adv}")

    def restore_training_weights(self):
        """Restore the training-time curriculum weight."""
        if hasattr(self, '_training_w_adv'):
            self.current_w_adv = self._training_w_adv
            del self._training_w_adv
            print(f"Restored training mode with w_adv = {self.current_w_adv}")

    def before_episode(self, env):
        self.env = env
        self._internal_adv_generator.before_episode(env)

    def log_AV_history(self):
        """Record ego-vehicle history for closed-loop generation."""
        self._internal_adv_generator.log_AV_history()

    def after_episode(self, update_AV_traj=False, mode='train'):
        """Update the stored ego-vehicle trajectory after an episode."""
        self._internal_adv_generator.after_episode(update_AV_traj=update_AV_traj, mode=mode)

    @property
    def adv_agent(self):
        return self._internal_adv_generator.storage.get(self.env.current_seed, {}).get('adv_agent')

    def get_data_for_scenario(self, env, mode='train'):
        """Collect model inputs, scene metadata, and the ego trajectory."""
        storage_entry = self._internal_adv_generator.storage[env.current_seed]
        traffic_motion_feat = storage_entry['traffic_motion_feat']
        batch_data = process_data(traffic_motion_feat, self.adversarial_model.args)
        adv_info = storage_entry['adv_info']
        adv_past_traj = storage_entry['adv_past']
        ego_info = storage_entry['ego_info']
        raw_map_features = env.engine.data_manager.get_scenario(env.current_seed)['map_features']

        ego_traj_deque = storage_entry.get('AV_trajs_eval' if mode == 'eval' else 'AV_trajs')

        if ego_traj_deque and len(ego_traj_deque) > 0:
            latest_ego_traj = ego_traj_deque[-1]
            print(f"[{mode.upper()} mode] Using latest RL agent trajectory for generation.")
        else:
            print(f"[{mode.upper()} mode] RL agent trajectory not available, falling back to GT trajectory.")
            ego_gt_future_traj_x = traffic_motion_feat['state/future/x'].numpy()[0, :, np.newaxis]
            ego_gt_future_traj_y = traffic_motion_feat['state/future/y'].numpy()[0, :, np.newaxis]
            latest_ego_traj = np.concatenate([ego_gt_future_traj_x, ego_gt_future_traj_y], axis=-1)

        return batch_data, traffic_motion_feat, adv_info, ego_info, latest_ego_traj, adv_past_traj, raw_map_features

    def generate(self, mode='train'):
        """
        Generate a SAGE adversarial trajectory from recent RL-agent behavior.

        The mode selects either the training or evaluation ego-trajectory buffer.
        """
        try:
            batch_data, traffic_motion_feat, adv_info, ego_info, ego_agent_future_traj, adv_past_traj, raw_map_features = self.get_data_for_scenario(
                self.env, mode=mode)
        except (KeyError, IndexError) as e:
            print(
                f"Warning: Data for seed {self.env.current_seed} not found or invalid. Error: {e}. Skipping generation.")
            self.adv_traj = []
            return

        w_real = 1.0 - self.current_w_adv
        souped_vectornet_model = create_souped_model(
            self.adversarial_model.model, self.realism_model.model,
            self.current_w_adv, w_real, self.device
        )
        temp_souped_model_wrapper = deepcopy(self.adversarial_model)
        temp_souped_model_wrapper.model = souped_vectornet_model

        pred_trajs_list, _, _ = temp_souped_model_wrapper.model(batch_data[0], self.device, return_tensors_for_hgpo=True)
        adv_candidate_trajs_np = pred_trajs_list[1]
        rewards = []
        for traj in adv_candidate_trajs_np:
            realism_penalties = calculate_realism_penalty(traj, adv_info)
            total_realism_penalty = (realism_penalties["behavior_penalty"] + realism_penalties["kinematic_penalty"])

            adversarial_rew, is_collision = calculate_adversarial_reward(traj, ego_agent_future_traj, adv_info,
                                                                         ego_info)

            map_violations = calculate_map_violation_penalty(traj, raw_map_features, traffic_motion_feat, adv_info)
            total_map_penalty = sum(map_violations.values())
            total_reward = self.current_w_adv * adversarial_rew - w_real * total_realism_penalty - total_map_penalty
            rewards.append(total_reward)

        winner_idx = np.argmax(rewards)
        adv_winner_traj_future = adv_candidate_trajs_np[winner_idx]

        adv_pos = np.concatenate((adv_past_traj, adv_winner_traj_future), axis=0)
        adv_yaw = get_polyline_yaw(adv_pos).reshape(-1, 1)
        adv_vel = get_polyline_vel(adv_pos)
        self.adv_traj = list(np.concatenate((adv_pos, adv_vel, adv_yaw), axis=1))

        return None, self.adv_traj, None, True


def eval_policy(policy, eval_env, adv_generator, eval_episodes=70, eval_w_adv=0.5):
    with torch.no_grad():
        _rewards_rc, _costs_coll, _rewards, _costs = [], [], [], []
        print(f"--- Starting Evaluation (Normal Scenarios to collect agent behavior) ---")
        for ep_num in range(eval_episodes):
            try:
                state, done = eval_env.reset(), False
                adv_generator.before_episode(eval_env)
                episode_reward = 0
                episode_cost = 0
                while not done:
                    adv_generator.log_AV_history()
                    action = policy.select_action(np.array(state))
                    state, reward, done, info = eval_env.step(action)
                    episode_reward += reward
                    episode_cost += info['cost']
                    if info.get('crash_vehicle', False): break
                _rewards_rc.append(info['route_completion'])
                _costs_coll.append(float(info.get('crash_vehicle', False)))
                _rewards.append(episode_reward)
                _costs.append(episode_cost)
                adv_generator.after_episode(update_AV_traj=True, mode='eval')
            except Exception as e:
                print(f"Warning: normal evaluation episode {ep_num} failed: {e}")
                _rewards.append(0.0)
                _costs.append(1.0)

        avg_reward_rc_normal = np.mean(_rewards_rc)
        avg_reward_normal = np.mean(_rewards)
        avg_cost_coll_normal = np.mean(_costs_coll)
        avg_cost_normal = np.mean(_costs)

        _rewards_rc, _costs_coll, _rewards, _costs = [], [], [], []
        print(f"--- Starting Evaluation (Adversarial Scenarios) ---")
        adv_generator.set_evaluation_weights(w_adv=eval_w_adv)
        for ep_num in range(eval_episodes):
            try:
                state, done = eval_env.reset(), False
                adv_generator.before_episode(eval_env)
                adv_generator.generate(mode='eval')
                eval_env.engine.traffic_manager.set_adv_info(adv_generator.adv_agent, adv_generator.adv_traj)
                episode_reward = 0
                episode_cost = 0
                while not done:
                    action = policy.select_action(np.array(state))
                    state, reward, done, info = eval_env.step(action)
                    episode_reward += reward
                    episode_cost += info['cost']
                    if info.get('crash_vehicle', False): break
                _rewards_rc.append(info['route_completion'])
                _costs_coll.append(float(info.get('crash_vehicle', False)))
                _rewards.append(episode_reward)
                _costs.append(episode_cost)
            except Exception as e:
                print(f"Warning: adversarial evaluation episode {ep_num} failed: {e}")
                _rewards.append(0.0)
                _costs.append(1.0)

        adv_generator.restore_training_weights()
        avg_reward_rc_adv = np.mean(_rewards_rc)
        avg_reward_adv = np.mean(_rewards)
        avg_cost_coll_adv = np.mean(_costs_coll)
        avg_cost_adv = np.mean(_costs)


    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes:")
    print(f"\tNormal Scenarios: Avg Reward: {avg_reward_normal:.3f}, Crash Rate: {avg_cost_normal:.3f}")
    print(f"\tAdversarial Scenarios: Avg Reward: {avg_reward_adv:.3f}, Crash Rate: {avg_cost_adv:.3f}")
    print("---------------------------------------")
    return avg_reward_normal, avg_reward_rc_normal, avg_cost_normal, avg_cost_coll_normal, avg_reward_adv, avg_reward_rc_adv, avg_cost_adv, avg_cost_coll_adv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")
    parser.add_argument("--env", default="MDWaymo")
    parser.add_argument("--seed", default=97, type=int)
    parser.add_argument("--start_timesteps", default=10000, type=int)
    parser.add_argument("--eval_freq", default=50000, type=int)
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--expl_noise", default=0.1, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--policy_noise", default=0.2, type=float)
    parser.add_argument("--noise_clip", default=0.5, type=float)
    parser.add_argument("--policy_freq", default=2, type=int)
    parser.add_argument("--save_model", default=True)
    parser.add_argument("--load_model", default="")

    parser.add_argument('--adversarial_model_path', type=str,
                        default='./advgen/finetuned/hgpo_finetuned_model_adv_best.bin',
                        help="Path to the HGPO fine-tuned adversarial model.")
    parser.add_argument('--realism_model_path', type=str,
                        default='./advgen/finetuned/hgpo_finetuned_model_real_best.bin',
                        help="Path to the original pre-trained model for realism.")
    parser.add_argument('--initial_w_adv', type=float, default=0.5,
                        help="Initial weight for the adversarial objective in curriculum learning.")
    parser.add_argument('--final_w_adv', type=float, default=1.0,
                        help="Final weight for the adversarial objective in curriculum learning.")
    parser.add_argument('--eval_w_adv', type=float, default=1.0,
                        help="Fixed adversarial weight for evaluation scenarios.")
    parser.add_argument('--min_prob', type=float, default=0.1, help="Min probability of using adv data in curriculum")
    parser.add_argument('--data_directory', type=str, default='./raw_scenes_370',
                        help="Path to processed MetaDrive/WOMD scenarios for RL training.")
    parser.add_argument('--train_scenarios', type=int, default=300)
    parser.add_argument('--eval_scenarios', type=int, default=70)

    args = parser.parse_args()

    # --- Setup ---
    file_name = "SAGE_TD3_ClosedLoop"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}, Method: SAGE (Closed-Loop)")
    print("---------------------------------------")
    print("Initializing SAGE adversarial generator for RL...")
    adv_generator = SAGEAdvGeneratorForRL(parser, sage_args=args)

    config_train = dict(
        data_directory=args.data_directory,
        start_scenario_index=0,
        num_scenarios=args.train_scenarios,
        sequential_seed=False,
        force_reuse_object_name=True,
        horizon=50,
        no_light=True,
        no_static_vehicles=True,
        reactive_traffic=False,
        vehicle_config=dict(
            lidar=dict(num_lasers=30, distance=50, num_others=3),
            side_detector=dict(num_lasers=30),
            lane_line_detector=dict(num_lasers=12)),
    )
    config_test = dict(
        data_directory=args.data_directory,
        start_scenario_index=args.train_scenarios,
        num_scenarios=args.eval_scenarios,
        crash_vehicle_done=True,
        sequential_seed=True,
        force_reuse_object_name=True,
        horizon=50,
        no_light=True,
        no_static_vehicles=True,
        reactive_traffic=False,
        vehicle_config=dict(
            lidar=dict(num_lasers=30, distance=50, num_others=3),
            side_detector=dict(num_lasers=30),
            lane_line_detector=dict(num_lasers=12)),
    )
    env = WaymoEnv(config=config_train)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        cudnn.deterministic = True
        cudnn.benchmark = False

    logger = SafeLogger(exp_name=file_name, env_name=args.env, seed=args.seed,
                        fieldnames=['route_completion_normal', 'crash_rate_normal', 'route_completion_adv',
                                    'crash_rate_adv'])
    writer = SummaryWriter(logger.log_dir)
    os.makedirs(f"{logger.log_dir}/models")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    kwargs = {
        "state_dim": state_dim, "action_dim": action_dim, "max_action": max_action,
        "discount": args.discount, "tau": args.tau,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq
    }
    policy = TD3.TD3(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # --- Main Training Loop ---
    scenario_random_seed = np.random.randint(0, 300)
    state, done = env.reset(force_seed=scenario_random_seed), False
    adv_generator.before_episode(env)

    episode_reward = 0
    episode_cost = 0
    episode_timesteps = 0
    episode_num = 0
    last_eval_step = 0

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        adv_generator.log_AV_history()

        adv_generator.update_weights(t, args.max_timesteps)

        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (policy.select_action(np.array(state)) +
                      np.random.normal(0, float(env.action_space.high[0]) * args.expl_noise,
                                       size=env.action_space.shape[0])
                      ).clip(-env.action_space.high[0], env.action_space.high[0])

        next_state, reward, done, info = env.step(action)
        replay_buffer.add(state, action, next_state, reward, float(done))
        crash = env.vehicle.crash_vehicle
        state = next_state
        episode_reward += reward
        episode_cost += info['cost']

        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            adv_generator.after_episode(update_AV_traj=True, mode='train')

            writer.add_scalar('Train/episode_reward', episode_reward, t)
            writer.add_scalar('Train/route_completion', info['route_completion'], t)
            if info['out_of_road']:
                writer.add_scalar('Train/out_of_road', 1, t)
            else:
                writer.add_scalar('Train/out_of_road', 0, t)
            if crash:
                writer.add_scalar('Train/crash', 1, t)
            else:
                writer.add_scalar('Train/crash', 0, t)
            print(
                f"Total T: {t + 1} | Episode {episode_num + 1} | Steps: {episode_timesteps} | Reward: {episode_reward:.3f} | Cost: {episode_cost:.3f}")

            if t - last_eval_step >= args.eval_freq:
                last_eval_step = t
                print("\n--- Running Evaluation ---")
                env.close()
                eval_env = WaymoEnv(config=config_test)
                eval_env.seed(args.seed)
                eval_env.action_space.seed(args.seed)
                evalReward_normal, evalRC_normal, evalCost_normal, evalCrash_normal, evalReward_adv, evalRC_adv, evalCost_adv, evalCrash_adv = eval_policy(policy, eval_env,
                                                                                         adv_generator,
                                                                                         eval_w_adv=args.eval_w_adv)
                eval_env.close()
                env = WaymoEnv(config=config_train)
                env.seed(args.seed)
                env.action_space.seed(args.seed)

                logger.update([evalRC_normal, evalCrash_normal, evalRC_adv, evalCrash_adv], total_steps=t + 1)
                if args.save_model:
                    policy.save(f"{logger.log_dir}/models/{file_name}")
                writer.add_scalar('Evaluate/route_completion_normal', evalRC_normal, t + 1)
                writer.add_scalar('Evaluate/crash_rate_normal', evalCrash_normal, t + 1)
                writer.add_scalar('Evaluate/route_completion_adv_1_ooi', evalRC_adv, t + 1)
                writer.add_scalar('Evaluate/crash_rate_adv_1_ooi', evalCrash_adv, t + 1)
                writer.add_scalar('Evaluate/reward_normal', evalReward_normal, t + 1)
                writer.add_scalar('Evaluate/cost_normal', evalCost_normal, t + 1)
                writer.add_scalar('Evaluate/reward_adv_1_ooi', evalReward_adv, t + 1)
                writer.add_scalar('Evaluate/cost_adv_1_ooi', evalCost_adv, t + 1)
                print("--- Evaluation Finished ---\n")

            try:
                scenario_random_seed = np.random.randint(0, 300)
                state, done = env.reset(force_seed=scenario_random_seed), False
                adv_generator.before_episode(env)
            except Exception as e:
                print(f"Error resetting env, trying again. Error: {e}")
                scenario_random_seed = np.random.randint(0, 300)
                state, done = env.reset(force_seed=scenario_random_seed), False
                adv_generator.before_episode(env)

            writer.add_scalar('Scenario_seed/seed', scenario_random_seed, t + 1)
            env.engine.traffic_manager.adv_name = []
            env.engine.traffic_manager.adv_traj = []

            if np.random.random() > max(1-(2*t/args.max_timesteps)*(1-args.min_prob),args.min_prob):
                print(
                    f'>>> Generating SAGE Adversarial Scenario (w_adv={adv_generator.current_w_adv:.3f}) <<<')
                adv_generator.generate(mode='train')
                writer.add_scalar('Scenario_seed/adv', 1, t+1)
            else:
                print(f'>>> Generating Normal Scenario <<<')
                writer.add_scalar('Scenario_seed/adv', 0, t+1)
                adv_generator.adv_traj = []

            env.engine.traffic_manager.set_adv_info(adv_generator.adv_agent, adv_generator.adv_traj)

            episode_reward, episode_cost, episode_timesteps = 0, 0, 0
            episode_num += 1

    env.close()
