"""Show a trained walking policy in the PyBullet GUI.

Default loads:
- stable_walk.zip
- stable_walk_vecnorm.pkl

Run:
  .\venv\Scripts\Activate.ps1
  python show_stable_walk.py
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environments import TrotWalkEnv


def make_env(gui: bool):
    return TrotWalkEnv(gui=gui, max_episode_steps=1200, action_scale=0.10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="stable_walk")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=1.0 / 240.0)
    args = ap.parse_args()

    model_path = args.model + ".zip"
    stats_path = args.model + "_vecnorm.pkl"
    if not os.path.isfile(model_path):
        raise SystemExit(f"Missing {model_path} — run train_stable_walk.py first.")
    if not os.path.isfile(stats_path):
        raise SystemExit(f"Missing {stats_path} — re-train or copy {stats_path}.")

    venv = DummyVecEnv([lambda: make_env(gui=True)])
    venv = VecNormalize.load(stats_path, venv)
    venv.training = False
    venv.norm_reward = False

    model = PPO.load(args.model, env=venv)

    for ep in range(args.episodes):
        obs = venv.reset()
        ret = 0.0
        for _ in range(2000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            ret += float(reward[0])
            time.sleep(args.sleep)
            if bool(done[0]):
                break
        print(f"episode {ep + 1}: return={ret:.2f}")

    venv.close()


if __name__ == "__main__":
    main()

