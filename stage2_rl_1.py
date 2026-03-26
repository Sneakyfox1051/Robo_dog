"""Alternate stage 2: continue walk_stage1 with smaller batches."""

from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO

from environments import TrotWalkEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", type=str, default="walk_stage1")
    ap.add_argument("--timesteps", type=int, default=80_000)
    ap.add_argument("--save", type=str, default="stage2_model_1")
    args = ap.parse_args()

    load_path = args.load + ".zip"
    if not os.path.isfile(load_path):
        raise SystemExit(f"Missing {load_path} — run stage1_walk_rl.py first.")

    env = TrotWalkEnv()
    model = PPO.load(load_path, env=env)
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False)
    model.save(args.save)
    env.close()
    print(f"stage2_rl_1.py: saved {args.save}.zip")


if __name__ == "__main__":
    main()
