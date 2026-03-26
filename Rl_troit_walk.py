"""PPO training for forward trot / walk (saved as Laikago imitation checkpoint name)."""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from environments import TrotWalkEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument(
        "--save",
        type=str,
        default="laikago_imitation_v1",
        help="SB3 saves <name>.zip — default matches project summary file.",
    )
    args = ap.parse_args()

    env = TrotWalkEnv()
    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=256)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save)
    env.close()
    print(f"Rl_troit_walk.py: saved {args.save}.zip")


if __name__ == "__main__":
    main()
