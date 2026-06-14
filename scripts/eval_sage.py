import argparse
import numpy as np
import os
import sys
import logging
from tqdm import trange, tqdm
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
from metadrive.policy.idm_policy import WaymoIDMPolicy
from metadrive_autopilot_safe import MetaDriveAutoPilotPolicy
from saferl_algo import TD3
# AdvGen Imports
from advgen.modeling.vectornet import VectorNet
from advgen.adv_utils import process_data
import advgen.utils as advgen_utils
from advgen.adv_generator import AdvGenerator as OriginalAdvGenerator  # For internal data parsing
from scipy.stats import wasserstein_distance
from sage.metrics import calculate_distributional_realism, get_kinematic_profiles
from sage.splits import scenario_ids

# ==========================================================================================
#  鏍稿績杈呭姪鍑芥暟 (鏁村悎鑷墍鏈夋彁渚涚殑鑴氭湰)
# ==========================================================================================
def moving_average(data, window_size):
    """骞虫粦鏁版嵁搴忓垪銆?"""
    interval = np.pad(data, window_size // 2, 'edge')
    window = np.ones(int(window_size)) / float(window_size)
    res = np.convolve(interval, window, 'valid')
    return res


def get_polyline_yaw(polyline):
    """璁＄畻杞ㄨ抗姣忎釜鐐圭殑鏈濆悜瑙掞紙yaw锛夈€?"""
    if polyline.shape[0] < 2:
        return np.zeros(polyline.shape[0])
    polyline_post = np.roll(polyline, shift=-1, axis=0)
    diff = polyline_post - polyline
    polyline_yaw = np.arctan2(diff[:, 1], diff[:, 0])
    polyline_yaw[-1] = polyline_yaw[-2]
    # 澶勭悊瑙掑害璺冲彉
    for i in range(len(polyline_yaw) - 1):
        if polyline_yaw[i + 1] - polyline_yaw[i] > 1.5 * np.pi:
            polyline_yaw[i + 1] -= 2 * np.pi
        elif polyline_yaw[i] - polyline_yaw[i + 1] > 1.5 * np.pi:
            polyline_yaw[i + 1] += 2 * np.pi
    return moving_average(polyline_yaw, window_size=5)


def get_polyline_vel(polyline):
    """鏍规嵁浣嶇Щ鍜屾椂闂存闀匡紙0.1s锛夎绠楅€熷害銆?"""
    polyline_post = np.roll(polyline, shift=-1, axis=0)
    polyline_post[-1] = polyline[-1]  # 鏈€鍚庝竴涓偣鐨勯€熷害涓庡墠涓€涓偣鐩稿悓
    diff = polyline_post - polyline
    polyline_vel = diff / 0.1
    return polyline_vel


def Intersect(l1, l2):
    """鍒ゆ柇涓ゆ潯绾挎鏄惁鐩镐氦锛堢敤浜庣鎾炴娴嬶級銆?"""
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


# ==========================================================================================
#  MOD鎺ㄧ悊鐨勬牳蹇冪被
# ==========================================================================================

class MotionModel:
    """涓€涓寘瑁呭櫒锛岀敤浜庡姞杞藉拰浣跨敤DenseTNT妯″瀷杩涜鎺ㄧ悊锛屽寘鍚笂涓嬫枃澶勭悊閫昏緫銆?"""

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
        """涓烘暣涓満鏅紙Ego+Adv锛夎绠楀叡浜殑涓婁笅鏂囪〃绀恒€?"""
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
        """鑾峰彇鏀诲嚮杞︼紙adv, index=1锛夌殑鍊欓€夌洰鏍囩偣鐨勫垎鏁帮紙log-probabilities锛夈€?"""
        all_mappings, merged_inputs_full, hidden_states_full, inputs_lengths_full = self._get_full_context(batch_data)

        # 纭繚鍦烘櫙涓湁鏀诲嚮杞﹁締
        if len(all_mappings) < 2:
            raise ValueError("MOD requires at least two agents (ego and adv) in the scene.")

        adv_mapping = all_mappings[1]
        goals_2D_tensor_adv = torch.tensor(adv_mapping['goals_2D'], device=self.device, dtype=torch.float)

        scores = self.model.decoder.get_scores(
            goals_2D_tensor_adv,
            merged_inputs_full, hidden_states_full, inputs_lengths_full,
            i=1,  # 鎸囧畾涓烘壒娆′腑鐨勭2涓猘gent锛堟敾鍑昏溅锛夎绠楀垎鏁?
            mapping=all_mappings, device=self.device
        )
        return scores

    @torch.no_grad()
    def generate_trajectories_for_goals(self, batch_data: list, top_k_goals: np.ndarray) -> np.ndarray:
        """涓虹粰瀹氱殑Top-K鐩爣鐐圭敓鎴愭渶缁堣建杩广€?"""
        all_mappings, merged_inputs_full, hidden_states_full, inputs_lengths_full = self._get_full_context(batch_data)

        if len(all_mappings) < 2:
            raise ValueError("MOD requires at least two agents (ego and adv) in the scene.")
        adv_mapping = all_mappings[1]

        k = len(top_k_goals)
        goals_2D_tensor_topk = torch.tensor(top_k_goals, device=self.device, dtype=torch.float)
        targets_feature_topk = self.model.decoder.goals_2D_mlps(goals_2D_tensor_topk)

        # 鏀诲嚮杞︼紙绱㈠紩1锛夌殑涓婁笅鏂?
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


def create_souped_model(model1, model2, w1, w2, device):
    """
    閫氳繃瀵逛袱涓ā鍨嬬殑鏉冮噸杩涜绾挎€ф彃鍊硷紝鍒涘缓涓€涓柊鐨勮瀺鍚堟ā鍨嬶紙Rewarded Soup锛夈€?

    Args:
        model1: 绗竴涓ā鍨?(e.g., adversarial_model.model)銆?
        model2: 绗簩涓ā鍨?(e.g., realism_model.model)銆?
        w1 (float): 绗竴涓ā鍨嬬殑鏉冮噸銆?
        w2 (float): 绗簩涓ā鍨嬬殑鏉冮噸銆?
        device: 妯″瀷鎵€鍦ㄧ殑璁惧銆?

    Returns:
        涓€涓叏鏂扮殑銆佹潈閲嶈瀺鍚堝悗鐨勬ā鍨嬪疄渚嬨€?
    """
    # 鍒涘缓涓€涓柊妯″瀷浣滀负瀹瑰櫒锛屼互鍏嶄慨鏀瑰師濮嬫ā鍨?
    # 鎴戜滑娣卞害鎷疯礉鍏朵腑涓€涓ā鍨嬬殑缁撴瀯鍜屽弬鏁颁綔涓鸿捣鐐?
    souped_model = deepcopy(model1)
    souped_model.to(device)

    # 鑾峰彇涓や釜妯″瀷鐨勫弬鏁板瓧鍏?
    params1 = model1.state_dict()
    params2 = model2.state_dict()

    # 鍒涘缓鏂扮殑铻嶅悎鍙傛暟瀛楀吀
    souped_params = souped_model.state_dict()

    # 閬嶅巻鎵€鏈夊弬鏁板苟杩涜鍔犳潈骞冲潎
    for name in souped_params.keys():
        if name in params1 and name in params2:
            souped_params[name].data.copy_(w1 * params1[name].data + w2 * params2[name].data)

    # 灏嗚瀺鍚堝悗鐨勫弬鏁板姞杞藉埌鏂版ā鍨嬩腑
    souped_model.load_state_dict(souped_params)
    return souped_model


from sage.model_soup import create_souped_model  # noqa: E402


class SAGEAdvGeneratorForEval:
    def __init__(self, adv_model_path, real_model_path, mod_args):
        self.mod_args = mod_args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading SAGE models...")
        self.adversarial_model = MotionModel(adv_model_path, self.device)
        self.realism_model = MotionModel(real_model_path, self.device)
        print(f"Using device: {self.device}")

        # 鍐呴儴浣跨敤鍘熷鐢熸垚鍣ㄦ潵澶嶇敤鍏跺鏉傜殑鏁版嵁瑙ｆ瀽鍜屽瓨鍌ㄩ€昏緫
        # 娉ㄦ剰锛氳繖閲岀殑parser鍙傛暟鍙互涓虹┖锛屽洜涓哄畠鍙敤浜庤缃唴閮╝dv_generator鐨勯粯璁ゅ€?
        _parser = argparse.ArgumentParser()
        _parser.add_argument('--OV_traj_num', type=int, default=32)
        _parser.add_argument('--AV_traj_num', type=int, default=1)
        self._internal_adv_generator = OriginalAdvGenerator(_parser)
        self.adv_traj = None
        self.env = None

        # 鍏变韩涓€濂?AdvGen 鍙傛暟
        self.args = self.adversarial_model.args

    def before_episode(self, env):
        self.env = env
        # 澶嶇敤鍘熷鐢熸垚鍣ㄧ殑鍦烘櫙鏁版嵁瑙ｆ瀽鍜屽瓨鍌ㄥ垵濮嬪寲
        self._internal_adv_generator.before_episode(env)

    def log_AV_history(self):
        """鍦ㄦ瘡涓€姝ヨ褰曚富杞﹁建杩广€?"""
        # 杩欎釜鍑芥暟鐜板湪鍙槸涓€涓畝鍗曠殑鍖呰锛屽疄闄呭伐浣滅敱鍐呴儴鐢熸垚鍣ㄥ畬鎴愩€?
        self._internal_adv_generator.log_AV_history()

    def after_episode(self, update_AV_traj=False, mode='train',n=10):
        """鍦ㄤ竴涓洖鍚堢粨鏉熷悗锛屽鐞嗗苟瀛樺偍璁板綍涓嬬殑涓昏溅杞ㄨ抗銆?"""
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
        """澶嶇敤鍐呴儴鐢熸垚鍣ㄥ拰process_data鏉ヨ幏鍙栨墍鏈夐渶瑕佺殑鏁版嵁"""
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

        # <<< 闂幆閫昏緫鍏抽敭鐐?2 >>>: 浣跨敤RL鏅鸿兘浣撶殑鏈€鏂拌建杩癸紝鑰屼笉鏄暟鎹泦涓殑GT杞ㄨ抗
        # 鏍规嵁鏄缁冭繕鏄瘎浼帮紝浠庝笉鍚岀殑deque涓幏鍙栬建杩?
        ego_traj_deque = self._internal_adv_generator.storage[env.current_seed].get('AV_trajs_eval' if mode == 'eval' else 'AV_trajs')

        # 浣跨敤deque涓渶鏂扮殑杞ㄨ抗銆傚鏋渄eque涓虹┖锛堜緥濡傦紝鍦ㄧ涓€娆¤繍琛屾煇涓満鏅椂锛夛紝鍒欏洖閫€鍒颁娇鐢ㄥ師濮婫T杞ㄨ抗
        if ego_traj_deque and len(ego_traj_deque) > 0:
            # -1 绱㈠紩鑾峰彇鏈€鏂版坊鍔犵殑杞ㄨ抗
            latest_ego_traj = ego_traj_deque[-1]
            print(f"[{mode.upper()} mode] Using latest RL agent trajectory for generation.")
        else:
            # 鍥為€€鏂规锛氫娇鐢ㄦ暟鎹泦涓殑鐪熷€艰建杩?
            print(f"[{mode.upper()} mode] RL agent trajectory not available, falling back to GT trajectory.")
            ego_gt_future_traj_x = traffic_motion_feat['state/future/x'].numpy()[0, :, np.newaxis]
            ego_gt_future_traj_y = traffic_motion_feat['state/future/y'].numpy()[0, :, np.newaxis]
            latest_ego_traj = np.concatenate([ego_gt_future_traj_x, ego_gt_future_traj_y], axis=-1)

        return batch_data, traffic_motion_feat, adv_info, ego_info, latest_ego_traj, adv_past_traj, raw_map_features

    def generate_with_mod(self):
        """
        浣跨敤MOD鏂规硶鐢熸垚瀵规姉杞ㄨ抗锛屽苟绔嬪嵆杩涜璇勪及銆?
        杩斿洖: (鐢ㄤ簬妯℃嫙鐨勮建杩? 璇勪及鎸囨爣瀛楀吀)
        """
        # 1. 鑾峰彇褰撳墠鍦烘櫙鎵€闇€鐨勬墍鏈夋暟鎹?
        batch_data, traffic_motion_feat, adv_info, ego_info, ego_agent_future_traj, adv_past_traj, raw_map_features = self.get_data_for_scenario(
            self.env)

        # 3. MOD铻嶅悎: 缁撳悎鍒嗘暟閫夋嫨鏈€浣崇洰鏍囩偣
        w_adv = self.mod_args.w_adv
        w_real = 1.0 - w_adv

        # 姝ラ 2: 鍔ㄦ€佸垱寤鸿瀺鍚堝悗鐨勬ā鍨?(Rewarded Soups鎬濇兂)
        souped_vectornet_model = create_souped_model(
            self.adversarial_model.model,
            self.realism_model.model,
            w_adv,
            w_real,
            self.device)

        # 鍒涘缓涓€涓复鏃剁殑MotionModel鍖呰鍣ㄦ潵浣跨敤souped_model鐨勮В鐮佸姛鑳?
        temp_souped_model_wrapper = deepcopy(self.adversarial_model)  # 澶嶅埗缁撴瀯鍜宎rgs
        temp_souped_model_wrapper.model = souped_vectornet_model  # 鏇挎崲涓鸿瀺鍚堝悗鐨勬ā鍨?

        pred_trajs_list, _, _ = temp_souped_model_wrapper.model(batch_data[0], self.device, return_tensors_for_dpo=True)
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
        # ################################################################################

        # 5. 銆愬叧閿€戝湪杩斿洖鍓嶏紝璁＄畻鎵€鏈夎瘎浼版寚鏍?
        map_violations = calculate_map_violation_penalty(final_traj_points, raw_map_features, traffic_motion_feat,
                                                         adv_info)
        realism_penalties = calculate_realism_penalty(final_traj_points, adv_info)
        adversarial_rew, is_collision = calculate_adversarial_reward(final_traj_points, ego_agent_future_traj, adv_info,
                                                                     ego_info)

        # ======================== [淇敼] 鏂板鍒嗗竷鐪熷疄鎬ф寚鏍囪绠?=========================
        # 1. 鎻愬彇瀵规姉杞︾殑鐪熷疄(GT)鏈潵杞ㄨ抗銆俉aymo鏁版嵁涓紝绱㈠紩0鏄富杞?ego)锛岀储寮?閫氬父鏄鎶楄溅
        adv_gt_x = traffic_motion_feat['state/future/x'].numpy()[1, :, np.newaxis]
        adv_gt_y = traffic_motion_feat['state/future/y'].numpy()[1, :, np.newaxis]
        adv_gt_valid = traffic_motion_feat['state/future/valid'].numpy()[1, :]

        adv_gt_traj_full = np.concatenate([adv_gt_x, adv_gt_y], axis=-1)
        adv_gt_traj = adv_gt_traj_full[adv_gt_valid]  # 鍙娇鐢ㄦ湁鏁堢殑杞ㄨ抗鐐?

        # 2. 璁＄畻褰撳墠杞ㄨ抗鐨勮繍鍔ㄥ鐢诲儚
        current_gen_profiles = get_kinematic_profiles(final_traj_points)
        current_gt_profiles = get_kinematic_profiles(adv_gt_traj)
        # ==============================================================================

        metrics = {
            "collision": 1.0 if is_collision else 0.0,
            "adversarial_reward": adversarial_rew,
            **map_violations,
            **realism_penalties,
        }

        # 6. 鏍煎紡鍖栬建杩逛互渚汳etadrive妯℃嫙浣跨敤
        adv_pos = np.concatenate((adv_past_traj, final_traj_points), axis=0)
        adv_yaw = get_polyline_yaw(adv_pos).reshape(-1, 1)
        adv_vel = get_polyline_vel(adv_pos)
        self.adv_traj = list(np.concatenate((adv_pos, adv_vel, adv_yaw), axis=1))

        return self.adv_traj, metrics, current_gen_profiles, current_gt_profiles, traffic_motion_feat

def evaluate_metrics(env,adv_generator,adv_traj_from_gen,raw_map_features, traffic_motion_feat):
    # 1. 浠巗torage鑾峰彇璇勪及鎵€闇€淇℃伅
    storage_entry = adv_generator._internal_adv_generator.storage[env.current_seed]
    adv_info = storage_entry['adv_info']
    ego_info = storage_entry['ego_info']

    # 2. 鎻愬彇涓昏溅GT鏈潵杞ㄨ抗
    # ego_gt_future_traj_x = traffic_motion_feat['state/future/x'].numpy()[0, :, np.newaxis]
    # ego_gt_future_traj_y = traffic_motion_feat['state/future/y'].numpy()[0, :, np.newaxis]
    # ego_gt_future_traj = np.concatenate([ego_gt_future_traj_x, ego_gt_future_traj_y], axis=-1)
    ego_traj_deque = storage_entry.get('AV_trajs')
    # 浣跨敤deque涓渶鏂扮殑杞ㄨ抗銆傚鏋渄eque涓虹┖锛堜緥濡傦紝鍦ㄧ涓€娆¤繍琛屾煇涓満鏅椂锛夛紝鍒欏洖閫€鍒颁娇鐢ㄥ師濮婫T杞ㄨ抗
    if ego_traj_deque and len(ego_traj_deque) > 0:
        # -1 绱㈠紩鑾峰彇鏈€鏂版坊鍔犵殑杞ㄨ抗
        latest_ego_traj = ego_traj_deque[-1]
    else:
        raise ValueError
    # 3. 灏嗙敓鎴愮殑杞ㄨ抗鍒楄〃杞负Numpy鏁扮粍
    adv_traj_np = np.array(adv_traj_from_gen)[11:, :2]  # only the trajectory

    # 4. 璋冪敤鏂扮殑璇勪及鍑芥暟
    map_violations = calculate_map_violation_penalty(adv_traj_np, raw_map_features, traffic_motion_feat, adv_info)
    realism_penalties = calculate_realism_penalty(adv_traj_np, adv_info)
    adversarial_rew, is_collision = calculate_adversarial_reward(adv_traj_np, latest_ego_traj, adv_info, ego_info)

    # ======================== [淇敼] 鑱氬悎鍒嗗竷鐪熷疄鎬ф暟鎹?=========================
    # 1. 鎻愬彇瀵规姉杞︾殑鐪熷疄(GT)鏈潵杞ㄨ抗
    adv_gt_x = traffic_motion_feat['state/future/x'].numpy()[1, :, np.newaxis]
    adv_gt_y = traffic_motion_feat['state/future/y'].numpy()[1, :, np.newaxis]
    adv_gt_valid = traffic_motion_feat['state/future/valid'].numpy()[1, :]
    adv_gt_traj_full = np.concatenate([adv_gt_x, adv_gt_y], axis=-1)
    adv_gt_traj = adv_gt_traj_full[adv_gt_valid]


    # 5. 缁勫悎鎴愪竴涓畬鏁寸殑鎸囨爣瀛楀吀
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
    parser.add_argument('--adv_model_path', type=str, default='./advgen/finetuned/grpo_finetuned_model_adv_best.bin',
                        help="Path to the 'adversarial' fine-tuned model.")
    parser.add_argument('--real_model_path', type=str, default='./advgen/finetuned/grpo_finetuned_model_real_best.bin',
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

    # 鍒濆鍖朚OD瀵规姉鐢熸垚鍣?
    adv_generator = SAGEAdvGeneratorForEval(
        adv_model_path=args.adv_model_path,
        real_model_path=args.real_model_path,
        mod_args=args
    )

    extra_args = dict(mode="top_down", film_size=(2200, 2200), screen_size=(600, 600))

    # === 鍒濆鍖栫粺璁℃寚鏍?(瀹屽叏閬靛惊 baseline 鏍煎紡) ===
    time_cost = 0.
    results_list = []  # 瀛樺偍姣忎釜鍦烘櫙璇︾粏鎸囨爣鐨勫垪琛?
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

        # 瀵煎叆璁粌濂界殑TD3
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
            # "agent_policy": WaymoIDMPolicy,  
            "agent_policy": ReplayEgoCarPolicy,  
            # "agent_policy": MetaDriveAutoPilotPolicy,  
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

    # [鏂板] 鍒濆鍖栫敤浜庤仛鍚堝垎甯冩暟鎹殑鍒楄〃
    all_gen_profiles = {"vel": [], "acc": [], "yaw_rate": []}
    all_gt_profiles = {"vel": [], "acc": [], "yaw_rate": []}
    for i in pbar:
        ######################## 绗竴杞? 姝ｅ父鍥炴斁 (淇濇寔涓嶅彉) ########################
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
                # 濡傛灉璺緞涓嶅瓨鍦紝鍒涘缓璺緞
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

        ################ 绗簩杞? MOD瀵规姉鐢熸垚涓庤瘎浼?#####################
        o = env.reset(force_seed=i)
        env.vehicle.ego_crash_flag = False
        done = False

        t0 = time.time()
        # before_episode is called inside get_data_for_scenario, so we just call the main function
        # adv_generator.before_episode(env) # This is now handled internally

        # *** 鏍稿績鏀瑰姩: 璋冪敤鏂扮殑MOD鐢熸垚鍜岃瘎浼版柟娉?***
        mod_traj_for_sim, current_metrics, gen_profiles, gt_profiles, traffic_motion_feat = adv_generator.generate_with_mod()

        t1 = time.time()
        time_cost += t1 - t0

        if mod_traj_for_sim and current_metrics:
            # 1. 鑱氬悎杩愬姩瀛︾敾鍍忔暟鎹?
            # for key in all_gen_profiles.keys():
            #     all_gen_profiles[key].extend(gen_profiles[key])
            #     all_gt_profiles[key].extend(gt_profiles[key])
            # results_list.append(current_metrics)
            env.engine.traffic_manager.set_adv_info(adv_generator.adv_agent, mod_traj_for_sim)
            mod_adv_traj_for_sim = mod_traj_for_sim.copy()
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
                ret = env.render(**extra_args, text={'Baseline Generate': f'Safety-Critical Scenario, seed {i}'})
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
        current_metrics = evaluate_metrics(env,adv_generator,mod_adv_traj_for_sim,raw_map_features,traffic_motion_feat)
        for key in all_gen_profiles.keys():
            all_gen_profiles[key].extend(gen_profiles[key])
            all_gt_profiles[key].extend(gt_profiles[key])
        results_list.append(current_metrics)


        # === 鏇存柊杩涘害鏉★紝鏄剧ず涓?baseline 涓€鑷寸殑鑱氬悎鎸囨爣 ===
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

    # ========================= [淇敼] 鍦ㄥ惊鐜悗璁＄畻鏈€缁堢殑鑱氬悎WD鎸囨爣 =========================
    final_dist_metrics = {}
    if results_list:  # 纭繚鑷冲皯杩愯浜嗕竴涓満鏅?
        # 灏嗗垪琛ㄨ浆鎹负Numpy鏁扮粍
        for key in all_gen_profiles.keys():
            all_gen_profiles[key] = np.array(all_gen_profiles[key])
            all_gt_profiles[key] = np.array(all_gt_profiles[key])

        # 璁＄畻鑱氬悎鍒嗗竷鐨刉asserstein璺濈
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
    # ========================================================================================

    # === 鏈€缁堢粨鏋滀互涓?baseline 瀹屽叏鐩稿悓鐨勮〃鏍煎舰寮忔墦鍗?===
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
        # ========================= [淇敼] 灏嗚仛鍚圵D鎸囨爣娣诲姞鍒皊ummary_stats涓?=========================
        for key, value in final_dist_metrics.items():
            summary_stats[key] = value
        # ==========================================================================================
        num_scenarios = len(results_df)

        print(f"Total Scenarios Tested: {num_scenarios}")
        print(f"Average Generation Time: {time_cost / num_scenarios:.4f} s/scenario")
        print("-" * 70)

        # 1. 鎬讳綋鍜屽鎶楁€ф寚鏍?
        print(f"{'Overall & Adversarial Metrics':<45} | {'Value':<20}")
        print("-" * 70)
        print(f"{'Collision Rate':<45} | {summary_stats.get('collision', 0):.2%}")
        print(f"{'Adversarial Reward (瓒婇珮瓒婂ソ)':<45} | {summary_stats.get('adversarial_reward', 0):.4f}")
        print("-" * 70)

        # 2. 鐪熷疄鎬ф儵缃?(瓒婁綆瓒婂ソ)
        print(f"{'Realism Penalties (瓒婁綆瓒婂ソ)':<45} | {'Value':<20}")
        print("-" * 70)
        realism_keys = sorted(
            [k for k in summary_stats.index if 'penalty' in k and 'kinematic' in k or 'behavior' in k])
        for key in realism_keys:
            print(f"{key:<45} | {summary_stats.get(key, 0):.4f}")
        print("-" * 70)

        # ======================= [鏂板] 鎵撳嵃鍒嗗竷鐪熷疄鎬ф寚鏍?=========================
        print(f"{'Distributional Realism Metrics (WD, 瓒婁綆瓒婂ソ)':<45} | {'Value':<20}")
        print("-" * 70)
        dist_keys = sorted([
            'wd_velocity', 'wd_acceleration', 'wd_yaw_rate', 'realism_meta_metric'
        ])
        for key in dist_keys:
            print(f"{key:<45} | {summary_stats.get(key, 0):.4f}")
        print("-" * 70)
        # ============================================================================

        # 3. 鍦板浘鍚堣鎬ф儵缃?(瓒婁綆瓒婂ソ)
        print(f"{'Map Compliance Penalties (瓒婁綆瓒婂ソ)':<45} | {'Value':<20}")
        print("-" * 70)
        map_keys = sorted([k for k in summary_stats.index if 'map' in k or 'solid_line' in k or 'object' in k])
        if not map_keys:
            print(f"{'No map violation metrics found.':<45} | {'N/A':<20}")
        else:
            for key in map_keys:
                print(f"{key:<45} | {summary_stats.get(key, 0):.4f}")

    print("=" * 70)
