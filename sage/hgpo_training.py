"""HGPO fine-tuning utilities for SAGE expert models."""

from __future__ import annotations

import argparse
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import advgen.utils as advgen_utils
from advgen.adv_utils import process_data
from advgen.modeling.vectornet import VectorNet
from sage.rewards import (
    calculate_adversarial_reward,
    calculate_map_violation_penalty,
    calculate_realism_penalty,
)
from sage.splits import filter_ids_by_summary, scenario_ids


class HGPOAdvGenerator:
    """Motion forecasting model wrapper used by HGPO fine-tuning."""

    def __init__(self, parser: argparse.ArgumentParser, model_path: str):
        advgen_utils.add_argument(parser)
        parser.set_defaults(
            other_params=[
                "l1_loss",
                "densetnt",
                "goals_2D",
                "enhance_global_graph",
                "laneGCN",
                "point_sub_graph",
                "laneGCN-4",
                "stride_10_2",
                "raster",
                "train_pair_interest",
            ],
            mode_num=32,
            future_frame_num=80,
        )
        args = parser.parse_args([])

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
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
        if self._internal_adv_generator is None:
            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument("--OV_traj_num", type=int, default=32)
            parser.add_argument("--AV_traj_num", type=int, default=1)
            from advgen.adv_generator import AdvGenerator as OriginalAdvGenerator

            self._internal_adv_generator = OriginalAdvGenerator(parser)

        self._internal_adv_generator.before_episode(env)
        storage_entry = self._internal_adv_generator.storage[env.current_seed]
        traffic_motion_feat = storage_entry["traffic_motion_feat"]
        batch_data = process_data(traffic_motion_feat, self.args)
        adv_info = storage_entry["adv_info"]
        ego_info = storage_entry["ego_info"]
        ego_future_x = traffic_motion_feat["state/future/x"]
        ego_future_y = traffic_motion_feat["state/future/y"]
        if hasattr(ego_future_x, "numpy"):
            ego_future_x = ego_future_x.numpy()
        if hasattr(ego_future_y, "numpy"):
            ego_future_y = ego_future_y.numpy()
        ego_gt_future_traj = np.stack([ego_future_x[0, :], ego_future_y[0, :]], axis=-1)
        adv_past_traj = storage_entry["adv_past"]
        raw_map_features = env.engine.data_manager.get_scenario(env.current_seed)["map_features"]
        return (
            batch_data,
            traffic_motion_feat,
            adv_info,
            ego_info,
            ego_gt_future_traj,
            adv_past_traj,
            raw_map_features,
        )


def hgpo_loss(
    policy_log_probs_w: torch.Tensor,
    policy_log_probs_l: torch.Tensor,
    ref_log_probs_w: torch.Tensor,
    ref_log_probs_l: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    log_ratio_policy = policy_log_probs_w - policy_log_probs_l
    with torch.no_grad():
        log_ratio_ref = ref_log_probs_w - ref_log_probs_l
    return -F.logsigmoid(beta * (log_ratio_policy - log_ratio_ref))


def build_parser(
    *,
    default_adversarial_weight: float,
    default_realism_weight: float,
    default_save_path: str,
    default_run_name: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a DenseTNT motion model with HGPO preferences."
    )
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.05, help="Beta parameter for the HGPO loss.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--num_pairs_per_group",
        type=int,
        default=8,
        help="Number of preference pairs sampled from each candidate group.",
    )
    parser.add_argument(
        "--reward_margin",
        type=float,
        default=0.2,
        help="Minimum reward difference required for a preference pair.",
    )
    parser.add_argument(
        "--adversarial_weight",
        type=float,
        default=default_adversarial_weight,
        help="Weight for the adversarial reward in the HGPO preference score.",
    )
    parser.add_argument(
        "--realism_weight",
        type=float,
        default=default_realism_weight,
        help="Weight for the realism penalty in the HGPO preference score.",
    )
    parser.add_argument("--save_path", type=str, default=default_save_path)
    parser.add_argument("--base_model_path", type=str, default="./advgen/pretrained/densetnt.bin")
    parser.add_argument("--data_directory", type=str, default="./raw_scenes_500")
    parser.add_argument("--split_file", type=str, default="./configs/splits/sage_womd_500.json")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_scenarios", type=int, default=None)
    parser.add_argument(
        "--scenario_csv_path",
        type=str,
        default="./configs/splits/sage_autopilot_summary.csv",
        help="Scenario summary CSV used to filter released training scenarios.",
    )
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--run_name", type=str, default=default_run_name)
    return parser


def run_hgpo(args: argparse.Namespace) -> None:
    print("HGPO training arguments:", args)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_log_dir = os.path.join(args.log_dir, f"{args.run_name}_{timestamp}")
    writer = SummaryWriter(log_dir=run_log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ids = scenario_ids(args.split_file, split=args.split, max_scenarios=args.max_scenarios)
    train_ids = filter_ids_by_summary(train_ids, args.scenario_csv_path)
    print(f"Found {len(train_ids)} scenarios to train on after filtering.")
    if not train_ids:
        print("Error: No scenarios left after filtering. Please check split and summary files.")
        writer.close()
        return

    generator = HGPOAdvGenerator(argparse.ArgumentParser(), model_path=args.base_model_path)
    env = WaymoEnv(
        {
            "agent_policy": ReplayEgoCarPolicy,
            "reactive_traffic": False,
            "use_render": False,
            "data_directory": args.data_directory,
            "start_scenario_index": 0,
            "num_scenarios": max(train_ids) + 1,
            "force_reuse_object_name": True,
            "sequential_seed": True,
        }
    )

    optimizer = torch.optim.AdamW(generator.policy_model.parameters(), lr=args.learning_rate)
    all_train_ids = list(train_ids)
    global_step = 0
    best_avg_winner_reward = -float("inf")
    best_model_path = args.save_path.replace(".bin", "_best.bin")

    try:
        for epoch in range(args.epochs):
            print(f"\n--- Starting Epoch {epoch + 1}/{args.epochs} ---")
            random.shuffle(all_train_ids)

            epoch_metrics = {
                "total_loss": 0.0,
                "scenarios_processed": 0,
                "scenarios_skipped": 0,
                "pairs_processed": 0,
                "avg_winner_reward": [],
                "avg_loser_reward": [],
                "avg_winner_adv_rew": [],
                "collision_rate": [],
                "avg_winner_real_pen": [],
                "avg_winner_map_pen": [],
                "avg_winner_pen_behavior": [],
                "avg_winner_pen_kinematic": [],
                "avg_winner_pen_crash_object": [],
                "avg_winner_pen_cross_solid_line": [],
                "avg_feasibility_rate_all_candidates": [],
            }

            pbar = tqdm(all_train_ids, desc=f"Epoch {epoch + 1}")
            for seed in pbar:
                try:
                    env.reset(force_seed=seed)
                except Exception as exc:
                    print(f"Warning: failed to reset scenario {seed}: {exc}. Skipping.")
                    epoch_metrics["scenarios_skipped"] += 1
                    continue

                (
                    batch_data,
                    traffic_motion_feat,
                    adv_info,
                    ego_info,
                    ego_gt_future_traj,
                    _adv_past_traj,
                    raw_map_features,
                ) = generator.get_data_for_scenario(env)

                pred_trajs_list_np, pred_scores_list_t, _ = generator.policy_model(
                    batch_data[0], device, return_tensors_for_hgpo=True
                )
                adv_candidate_trajs_np = pred_trajs_list_np[1]
                adv_candidate_log_probs = pred_scores_list_t[1]

                trajectory_info = []
                for i, traj in enumerate(adv_candidate_trajs_np):
                    map_violations = calculate_map_violation_penalty(
                        traj, raw_map_features, traffic_motion_feat, adv_info
                    )
                    total_map_penalty = (
                        map_violations["cross_solid_line_penalty"]
                        + map_violations["crash_object_penalty"]
                    )
                    realism_penalties = calculate_realism_penalty(traj, adv_info)
                    total_realism_penalty = (
                        realism_penalties["behavior_penalty"]
                        + realism_penalties["kinematic_penalty"]
                    )
                    adversarial_reward, is_collision = calculate_adversarial_reward(
                        traj, ego_gt_future_traj, adv_info, ego_info
                    )
                    preference_reward = (
                        args.adversarial_weight * adversarial_reward
                        - args.realism_weight * total_realism_penalty
                    )

                    trajectory_info.append(
                        {
                            "index": i,
                            "is_feasible": total_map_penalty == 0,
                            "preference_reward": preference_reward,
                            "total_reward_for_log": preference_reward - total_map_penalty,
                            "adv_rew": adversarial_reward,
                            "is_collision": is_collision,
                            "real_pen_total": total_realism_penalty,
                            "map_pen_total": total_map_penalty,
                            **realism_penalties,
                            **map_violations,
                        }
                    )

                feasible_trajs = [info for info in trajectory_info if info["is_feasible"]]
                infeasible_trajs = [info for info in trajectory_info if not info["is_feasible"]]
                total_candidates = len(trajectory_info)
                epoch_metrics["avg_feasibility_rate_all_candidates"].append(
                    len(feasible_trajs) / total_candidates if total_candidates else 0.0
                )

                if len(feasible_trajs) < 1 or len(trajectory_info) < 2:
                    epoch_metrics["scenarios_skipped"] += 1
                    continue

                with torch.no_grad():
                    _, ref_scores_list, _ = generator.reference_model(
                        batch_data[0], device, return_tensors_for_hgpo=True
                    )
                    ref_log_probs = ref_scores_list[1]

                preference_pairs = []
                for loser_info in infeasible_trajs:
                    preference_pairs.append((random.choice(feasible_trajs), loser_info))

                remaining_budget = max(0, args.num_pairs_per_group - len(preference_pairs))
                if len(feasible_trajs) >= 2 and remaining_budget > 0:
                    for _ in range(remaining_budget):
                        first, second = random.sample(feasible_trajs, 2)
                        winner, loser = (
                            (first, second)
                            if first["preference_reward"] > second["preference_reward"]
                            else (second, first)
                        )
                        if winner["preference_reward"] - loser["preference_reward"] > args.reward_margin:
                            preference_pairs.append((winner, loser))

                if not preference_pairs:
                    epoch_metrics["scenarios_skipped"] += 1
                    continue

                scenario_total_loss = 0.0
                pairs_found = 0
                for winner_info, loser_info in preference_pairs:
                    winner_idx = winner_info["index"]
                    loser_idx = loser_info["index"]
                    pair_loss = hgpo_loss(
                        adv_candidate_log_probs[winner_idx],
                        adv_candidate_log_probs[loser_idx],
                        ref_log_probs[winner_idx].detach(),
                        ref_log_probs[loser_idx].detach(),
                        beta=args.beta,
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

                best_traj_for_log = max(feasible_trajs, key=lambda x: x["preference_reward"])
                worst_traj_for_log = min(trajectory_info, key=lambda x: x["total_reward_for_log"])

                epoch_metrics["total_loss"] += average_scenario_loss.item()
                epoch_metrics["scenarios_processed"] += 1
                epoch_metrics["pairs_processed"] += pairs_found
                epoch_metrics["avg_winner_reward"].append(best_traj_for_log["total_reward_for_log"])
                epoch_metrics["avg_loser_reward"].append(worst_traj_for_log["total_reward_for_log"])
                epoch_metrics["avg_winner_adv_rew"].append(best_traj_for_log["adv_rew"])
                epoch_metrics["collision_rate"].append(float(best_traj_for_log["is_collision"]))
                epoch_metrics["avg_winner_real_pen"].append(best_traj_for_log["real_pen_total"])
                epoch_metrics["avg_winner_map_pen"].append(best_traj_for_log["map_pen_total"])
                epoch_metrics["avg_winner_pen_behavior"].append(best_traj_for_log["behavior_penalty"])
                epoch_metrics["avg_winner_pen_kinematic"].append(best_traj_for_log["kinematic_penalty"])
                epoch_metrics["avg_winner_pen_crash_object"].append(
                    best_traj_for_log["crash_object_penalty"]
                )
                epoch_metrics["avg_winner_pen_cross_solid_line"].append(
                    best_traj_for_log["cross_solid_line_penalty"]
                )

                writer.add_scalar("Step/Loss", average_scenario_loss.item(), global_step)
                writer.add_scalar(
                    "Step/Reward/Best_Traj_Total_Reward",
                    best_traj_for_log["total_reward_for_log"],
                    global_step,
                )
                writer.add_scalar(
                    "Step/Reward/Best_Traj_Preference_Reward",
                    best_traj_for_log["preference_reward"],
                    global_step,
                )
                writer.add_scalar(
                    "Step/Feasibility/Is_Best_Traj_Feasible",
                    float(best_traj_for_log["is_feasible"]),
                    global_step,
                )
                writer.add_scalar(
                    "Step/Feasibility/Num_Feasible_Candidates",
                    len(feasible_trajs),
                    global_step,
                )
                global_step += 1

                pbar.set_postfix(
                    {
                        "loss": f"{average_scenario_loss.item():.3f}",
                        "best_rew": f"{best_traj_for_log['total_reward_for_log']:.2f}",
                        "feasible": f"{len(feasible_trajs)}/{len(trajectory_info)}",
                        "pairs": f"{pairs_found}",
                        "skip": epoch_metrics["scenarios_skipped"],
                    }
                )

            if epoch_metrics["scenarios_processed"] == 0:
                continue

            avg_loss = epoch_metrics["total_loss"] / epoch_metrics["scenarios_processed"]
            avg_win_rew = np.mean(epoch_metrics["avg_winner_reward"])
            avg_lose_rew = np.mean(epoch_metrics["avg_loser_reward"])
            avg_win_adv_rew = np.mean(epoch_metrics["avg_winner_adv_rew"])
            avg_win_real_pen = np.mean(epoch_metrics["avg_winner_real_pen"])
            avg_win_map_pen = np.mean(epoch_metrics["avg_winner_map_pen"])
            avg_coll_rate = np.mean(epoch_metrics["collision_rate"])
            avg_pairs_per_scenario = (
                epoch_metrics["pairs_processed"] / epoch_metrics["scenarios_processed"]
            )
            avg_feasibility_rate = np.mean(np.array(epoch_metrics["avg_winner_map_pen"]) == 0)
            avg_feasibility_all_candidates = np.mean(
                epoch_metrics["avg_feasibility_rate_all_candidates"]
            )

            if avg_win_rew > best_avg_winner_reward:
                best_avg_winner_reward = avg_win_rew
                print(
                    f"\n[Model Save] New best model at epoch {epoch + 1} "
                    f"with Avg Winner Reward: {avg_win_rew:.4f}"
                )
                print(f"  -> Feasibility Rate: {avg_feasibility_rate:.2%}")
                print(f"  -> Saving model to {best_model_path}")
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                torch.save(generator.policy_model.state_dict(), best_model_path)

            print(f"\n--- Epoch {epoch + 1} Summary ---")
            print(f"  Avg Loss: {avg_loss:.4f} | Avg Pairs per Scenario: {avg_pairs_per_scenario:.2f}")
            print(
                f"  Avg Best Traj Reward: {avg_win_rew:.2f} | "
                f"Avg Worst Traj Reward: {avg_lose_rew:.2f}"
            )
            print(
                f"  Best Traj Breakdown -> Adv Rew: {avg_win_adv_rew:.2f}, "
                f"Real Pen: {avg_win_real_pen:.2f}, Map Pen: {avg_win_map_pen:.2f}"
            )
            print(
                f"  Best Traj Collision Rate: {avg_coll_rate:.2%} | "
                f"Best Traj Feasibility Rate: {avg_feasibility_rate:.2%}"
            )
            print(f"  Avg Feasibility Rate (All Candidates): {avg_feasibility_all_candidates:.2%}")
            print(
                f"  Scenarios Processed: {epoch_metrics['scenarios_processed']} | "
                f"Skipped: {epoch_metrics['scenarios_skipped']}"
            )

            writer.add_scalar("Epoch/Loss", avg_loss, epoch)
            writer.add_scalar("Epoch/Metrics/Feasibility_Rate", avg_feasibility_rate, epoch)
            writer.add_scalar(
                "Epoch/Metrics/Feasibility_Rate_AllCandidates",
                avg_feasibility_all_candidates,
                epoch,
            )
            writer.add_scalar("Epoch/Metrics/Collision_Rate", avg_coll_rate, epoch)
            writer.add_scalar("Epoch/Metrics/Avg_Best_Traj_Total_Reward", avg_win_rew, epoch)
            writer.add_scalar("Epoch/Metrics/Avg_Pairs_per_Scenario", avg_pairs_per_scenario, epoch)
            writer.add_scalar(
                "Epoch/Avg_BestTraj_Components/1_Adversarial_Reward",
                avg_win_adv_rew,
                epoch,
            )
            writer.add_scalar(
                "Epoch/Avg_BestTraj_Components/2_Realism_Penalty",
                avg_win_real_pen,
                epoch,
            )
            writer.add_scalar(
                "Epoch/Avg_BestTraj_Components/3_Map_Penalty",
                avg_win_map_pen,
                epoch,
            )
            writer.add_scalar(
                "Epoch/Avg_Penalties/Behavior",
                np.mean(epoch_metrics["avg_winner_pen_behavior"]),
                epoch,
            )
            writer.add_scalar(
                "Epoch/Avg_Penalties/Kinematic",
                np.mean(epoch_metrics["avg_winner_pen_kinematic"]),
                epoch,
            )
            writer.add_scalar(
                "Epoch/Avg_Penalties/CrashObject",
                np.mean(epoch_metrics["avg_winner_pen_crash_object"]),
                epoch,
            )
            writer.add_scalar(
                "Epoch/Avg_Penalties/CrossSolidLine",
                np.mean(epoch_metrics["avg_winner_pen_cross_solid_line"]),
                epoch,
            )

        print(f"\nTraining finished. Saving final model to {args.save_path}")
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(generator.policy_model.state_dict(), args.save_path)
        print(f"Final model saved. Best model saved to {best_model_path}")
    finally:
        writer.close()
        env.close()


def main(
    *,
    default_adversarial_weight: float,
    default_realism_weight: float,
    default_save_path: str,
    default_run_name: str,
) -> None:
    parser = build_parser(
        default_adversarial_weight=default_adversarial_weight,
        default_realism_weight=default_realism_weight,
        default_save_path=default_save_path,
        default_run_name=default_run_name,
    )
    run_hgpo(parser.parse_args())
