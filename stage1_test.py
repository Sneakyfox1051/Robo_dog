"""Evaluate a trained balance policy (default: balance_model)."""

from __future__ import annotations

import argparse
import os

import numpy as np
from stable_baselines3 import PPO

from environments import DogBalanceEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="balance_model")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    mp = args.model + ".zip"
    if not os.path.isfile(mp):
        raise SystemExit(f"Missing {mp} — train with stage1_rl.py first.")

    env = DogBalanceEnv(gui=args.gui)
    model = PPO.load(args.model, env=env)
    rewards: list[float] = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(np.array(action))
            total += float(r)
            steps += 1
            done = term or trunc
        rewards.append(total)
        print(f"episode {ep + 1}: return={total:.2f} steps={steps}")
    print(f"mean return: {float(np.mean(rewards)):.2f}")
    env.close()


if __name__ == "__main__":
    main()
