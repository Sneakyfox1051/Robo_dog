"""Dog (Laikago) standing RL — PPO on balance environment."""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from environments import DogBalanceEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=150_000)
    ap.add_argument("--save", type=str, default="rl_dog_stand")
    args = ap.parse_args()

    env = DogBalanceEnv()
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=1024,
        batch_size=128,
        learning_rate=3e-4,
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save)
    env.close()
    print(f"rl.py: saved {args.save}.zip")


if __name__ == "__main__":
    main()
