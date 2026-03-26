"""Stage 1 walking emphasis: train on TrotWalkEnv from scratch."""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from environments import TrotWalkEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=220_000)
    ap.add_argument("--save", type=str, default="walk_stage1")
    args = ap.parse_args()

    env = TrotWalkEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save)
    env.close()
    print(f"stage1_walk_rl.py: saved {args.save}.zip")


if __name__ == "__main__":
    main()
