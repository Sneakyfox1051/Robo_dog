"""Stage 1: balance (DogBalanceEnv) → balance_model.zip."""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from environments import DogBalanceEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=180_000)
    ap.add_argument("--save", type=str, default="balance_model")
    args = ap.parse_args()

    env = DogBalanceEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save)
    env.close()
    print(f"stage1_rl.py: saved {args.save}.zip")


if __name__ == "__main__":
    main()
