"""Gymnasium standing / balance task with PPO."""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from environments import DogBalanceEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=150_000)
    ap.add_argument("--save", type=str, default="laikago_stand_model")
    args = ap.parse_args()

    env = DogBalanceEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save)
    env.close()
    print(f"RL_stand.py: saved {args.save}.zip")


if __name__ == "__main__":
    main()
