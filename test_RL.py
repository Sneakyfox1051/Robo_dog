"""Load PPO policy and run a few episodes (balance task)."""

from __future__ import annotations

import argparse
import os

import numpy as np
from stable_baselines3 import PPO

from environments import DogBalanceEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="balance_model")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.model + ".zip"):
        raise SystemExit(f"Missing {args.model}.zip — run stage1_rl.py or rl.py first.")

    env = DogBalanceEnv(gui=args.gui)
    model = PPO.load(args.model, env=env)
    for ep in range(args.episodes):
        obs, _ = env.reset()
        g = 0.0
        for _ in range(800):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(np.array(a))
            g += float(r)
            if term or trunc:
                break
        print(f"test_RL episode {ep + 1}: return={g:.2f}")
    env.close()


if __name__ == "__main__":
    main()
