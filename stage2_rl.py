"""Stage 2: fine-tune balance policy on walking rewards."""

from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO

from environments import TrotWalkEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", type=str, default="balance_model")
    ap.add_argument("--timesteps", type=int, default=120_000)
    ap.add_argument("--save", type=str, default="stage2_model")
    args = ap.parse_args()

    if not os.path.isfile(args.load + ".zip") and not os.path.isfile(args.load):
        raise SystemExit(
            f"Missing {args.load}.zip — run stage1_rl.py first."
        )

    env = TrotWalkEnv()
    path = args.load if args.load.endswith(".zip") else args.load + ".zip"
    model = PPO.load(path, env=env)
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False)
    model.save(args.save)
    env.close()
    print(f"stage2_rl.py: saved {args.save}.zip")


if __name__ == "__main__":
    main()
