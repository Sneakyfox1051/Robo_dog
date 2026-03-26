"""Train PPO on IK-augmented walking environment."""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from environments import IKAugmentedWalkEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=120_000)
    ap.add_argument("--save", type=str, default="rl_with_ik_model")
    args = ap.parse_args()

    env = IKAugmentedWalkEnv()
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=2.5e-4)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save)
    env.close()
    print(f"rl_with_ik.py: saved {args.save}.zip")


if __name__ == "__main__":
    main()
