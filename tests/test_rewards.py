import numpy as np

from sage.rewards import calculate_adversarial_reward, calculate_realism_penalty


def test_adversarial_reward_detects_overlap():
    adv = np.stack([np.linspace(0, 5, 20), np.zeros(20)], axis=1)
    ego = adv.copy()
    info = {"w": 2.0, "l": 4.0}
    reward, collision = calculate_adversarial_reward(adv, ego, info, info)
    assert collision
    assert reward > 0


def test_realism_penalty_is_finite():
    traj = np.stack([np.linspace(0, 8, 20), np.zeros(20)], axis=1)
    penalty = calculate_realism_penalty(traj, {"w": 2.0, "l": 4.0})
    assert np.isfinite(penalty["kinematic_penalty"])
    assert np.isfinite(penalty["behavior_penalty"])
